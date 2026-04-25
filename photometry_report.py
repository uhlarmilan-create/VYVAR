from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from decimal import Decimal, InvalidOperation


def generate_photometry_report(
    draft_dir: Path,
    obs_group: str,
    output_pdf: Path | None,
) -> Path | None:
    """
    Generuje PDF správu pre jednu noc pozorovania.
    Vracia cestu k vygenerovanému PDF.

    Ak knižnica reportlab nie je dostupná, vráti None (bez výnimky).
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
        from reportlab.platypus import Table, TableStyle
    except Exception as exc:  # noqa: BLE001
        logging.warning("reportlab nie je nainštalovaný, PDF preskakujem (%s)", exc)
        return None

    draft_dir = Path(draft_dir)
    obs_group = str(obs_group)

    platesolve_dir = draft_dir / "platesolve" / obs_group
    photometry_dir = platesolve_dir / "photometry"
    lc_dir = photometry_dir / "lightcurves"
    cache_dir = photometry_dir / "_report_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = photometry_dir / "photometry_summary.csv"
    comp_csv = photometry_dir / "comparison_stars_per_target.csv"
    at_csv_primary = platesolve_dir / "active_targets.csv"
    at_csv_alt = photometry_dir / "active_targets.csv"

    active_targets_csv = at_csv_primary if at_csv_primary.exists() else at_csv_alt

    if not summary_csv.exists():
        raise FileNotFoundError(f"Chýba photometry_summary.csv: {summary_csv}")

    summary_df = pd.read_csv(summary_csv, low_memory=False)
    comp_df = pd.read_csv(comp_csv, low_memory=False) if comp_csv.exists() else pd.DataFrame()
    at_df = pd.read_csv(active_targets_csv, low_memory=False) if active_targets_csv.exists() else pd.DataFrame()

    def _norm_cid(x: Any) -> str:
        s = str(x or "").strip()
        if not s or s.lower() in ("nan", "none"):
            return ""
        try:
            # Avoid float conversion (precision loss for large Gaia IDs in scientific notation).
            return str(int(Decimal(s)))
        except (InvalidOperation, ValueError, TypeError, OverflowError):
            try:
                return str(int(s))
            except Exception:
                return s

    if "catalog_id" in summary_df.columns:
        summary_df["_cid"] = summary_df["catalog_id"].map(_norm_cid)
    if "catalog_id" in at_df.columns:
        at_df["_cid"] = at_df["catalog_id"].map(_norm_cid)
    if "target_catalog_id" in comp_df.columns:
        comp_df["_tcid"] = comp_df["target_catalog_id"].map(_norm_cid)
    elif "catalog_id" in comp_df.columns:
        comp_df["_tcid"] = comp_df["catalog_id"].map(_norm_cid)

    # Join metadata (vsx_type, bp_rp) into summary for report ordering/labels.
    if not at_df.empty and "_cid" in at_df.columns and "_cid" in summary_df.columns:
        meta_cols = [c for c in ("vsx_type", "zone_flag", "bp_rp", "vsx_name") if c in at_df.columns]
        meta = at_df[["_cid"] + meta_cols].drop_duplicates("_cid")
        summary_df = summary_df.merge(meta, how="left", on="_cid", suffixes=("", "_at"))

    # Sorting: best lc_rms first.
    if "lc_rms" in summary_df.columns:
        summary_df["_lc_rms"] = pd.to_numeric(summary_df["lc_rms"], errors="coerce")
        summary_df = summary_df.sort_values("_lc_rms", ascending=True, na_position="last")

    def _obs_date_str() -> str:
        # Prefer MASTERSTAR DATE-OBS if present.
        ms_fits = platesolve_dir / "MASTERSTAR.fits"
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
            except Exception:
                pass
        return datetime.today().strftime("%d.%m.%Y")

    obs_date_human = _obs_date_str()
    date_token = datetime.today().strftime("%Y%m%d")
    try:
        # If obs_date_human is in YYYY-MM-DD, take it.
        if "-" in obs_date_human and len(obs_date_human) >= 10:
            date_token = obs_date_human.split("T", 1)[0].replace("-", "")
    except Exception:
        pass

    if output_pdf is None:
        date_str = datetime.today().strftime("%Y%m%d")
        output_pdf = (
            draft_dir
            / "platesolve"
            / obs_group
            / f"VYVAR_report_{obs_group}_{date_str}.pdf"
        )
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Styles / colors
    C_TITLE = colors.HexColor("#1a1a2e")
    C_GOOD = colors.HexColor("#2ecc71")
    C_MID = colors.HexColor("#f39c12")
    C_BAD = colors.HexColor("#e74c3c")

    def _metric_color(v: float) -> Any:
        if not np.isfinite(v):
            return colors.black
        if v < 0.05:
            return C_GOOD
        if v < 0.1:
            return C_MID
        return C_BAD

    # Aggregate metrics (cover page)
    n_lc = int(len(summary_df))
    med_rms = float(np.nanmedian(pd.to_numeric(summary_df.get("lc_rms"), errors="coerce"))) if n_lc else float("nan")
    rms_lt_005 = int((pd.to_numeric(summary_df.get("lc_rms"), errors="coerce") < 0.05).sum()) if n_lc else 0
    avg_good_comp = float(np.nanmean(pd.to_numeric(summary_df.get("n_good_comp"), errors="coerce"))) if n_lc else float("nan")
    best_rms = float(np.nanmin(pd.to_numeric(summary_df.get("lc_rms"), errors="coerce"))) if n_lc else float("nan")
    worst_rms = float(np.nanmax(pd.to_numeric(summary_df.get("lc_rms"), errors="coerce"))) if n_lc else float("nan")
    avg_bp_rp = float(np.nanmean(pd.to_numeric(summary_df.get("bp_rp"), errors="coerce"))) if n_lc else float("nan")
    setups = 1
    fwhm_px = float("nan")
    if (platesolve_dir / "MASTERSTAR.fits").exists():
        try:
            from ui_aperture_photometry import _load_fwhm  # local import

            fwhm_px = float(_load_fwhm(platesolve_dir / "MASTERSTAR.fits"))
        except Exception:
            pass
    aperture_px = float(np.nanmedian(pd.to_numeric(summary_df.get("aperture_px"), errors="coerce"))) if n_lc else float("nan")

    def _prepare_jpeg(src: Path, dst: Path, *, max_side_px: int = 1600, quality: int = 75) -> Path | None:
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

    def _plot_lightcurve_to_jpeg(lc_csv: Path, out_jpg: Path) -> Path | None:
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

            plt.figure(figsize=(10.5, 4.2), dpi=150)
            plt.scatter(x, y, s=6, c="#1f77b4", alpha=0.9, linewidths=0)
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.25)
            plt.xlabel(xcol)
            plt.ylabel(ycol)
            plt.tight_layout()
            out_png = out_jpg.with_suffix(".png")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_png, dpi=150)
            plt.close()
            # Convert to JPEG for size
            return _prepare_jpeg(out_png, out_jpg, max_side_px=1600, quality=75)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Plot lightcurve zlyhal (%s): %s", lc_csv, exc)
            return None

    # ---------------------------------------------------------------------
    # Canvas-based layout (precise positioning; one star per page)
    # ---------------------------------------------------------------------

    PAGE_W, PAGE_H = landscape(A4)
    M_LEFT = 1.0 * cm
    M_RIGHT = 1.0 * cm
    M_TOP = 1.0 * cm
    M_BOTTOM = 0.8 * cm
    USE_W = PAGE_W - M_LEFT - M_RIGHT
    USE_H = PAGE_H - M_TOP - M_BOTTOM

    # Star page geometry (cm -> pt)
    TITLE_H = 0.8 * cm
    METRICS_H = 0.5 * cm
    SEP_H = 0.1 * cm
    # GRAPH_H is dynamic per star (see draw_star_page)
    LC_W = 16.5 * cm
    GAP_W = 0.5 * cm
    FI_W = USE_W - LC_W - GAP_W

    NOTE_TXT = (
        "B-V vypočítané z Gaia BP-RP (Riello et al. 2021, ±0.05 mag) | "
        "Váhy COMP: w = 1/σ² — Broeg, Fernández & Neuhäuser, Astron. Nachr. 326, 134 (2005)"
    )

    def _page_footer(c: "canvas.Canvas") -> None:
        try:
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#1a1a2e"))
            c.drawRightString(PAGE_W - M_RIGHT, 0.45 * cm, f"Strana {c.getPageNumber()}")
            c.setFillColor(colors.black)
        except Exception:
            pass

    def _draw_image_fit(
        c: "canvas.Canvas",
        img_path: Path,
        x: float,
        y_top: float,
        w: float,
        h: float,
    ) -> None:
        """Draw image fitted into (w,h) box, aligned top-left."""
        try:
            if not img_path or not Path(img_path).exists():
                return
            ir = ImageReader(str(img_path))
            iw, ih = ir.getSize()
            if not iw or not ih:
                return
            sx = w / float(iw)
            sy = h / float(ih)
            s = min(sx, sy)
            dw = float(iw) * s
            dh = float(ih) * s
            # top-left alignment
            c.drawImage(ir, x, y_top - dh, width=dw, height=dh, mask="auto")
        except Exception:
            return

    def _draw_cover_page(c: "canvas.Canvas") -> None:
        y = PAGE_H - M_TOP
        # Logo (page 1 only)
        try:
            logo_path = (Path(__file__).resolve().parent / "img" / "VYVAR_logo.png").resolve()
        except Exception:  # noqa: BLE001
            logo_path = None
        if logo_path and Path(logo_path).exists():
            # Fit logo into a wide, short box near top-right.
            _draw_image_fit(
                c,
                logo_path,
                x=M_LEFT + USE_W - 8.0 * cm,
                y_top=PAGE_H - M_TOP + 0.2 * cm,
                w=8.0 * cm,
                h=2.2 * cm,
            )
        c.setFont("Helvetica-Bold", 22)
        c.setFillColor(C_TITLE)
        c.drawString(M_LEFT, y - 0.6 * cm, "VYVAR — Správa fotometrie")
        c.setFillColor(colors.black)

        y -= 1.4 * cm
        c.setFont("Helvetica", 11)
        c.drawString(M_LEFT, y, f"Draft:     {draft_dir.name}")
        y -= 0.6 * cm
        c.drawString(M_LEFT, y, f"Setup:     {obs_group}")
        y -= 0.6 * cm
        c.drawString(M_LEFT, y, f"Dátum:     {obs_date_human}")
        y -= 0.6 * cm
        c.drawString(M_LEFT, y, f"Vygenerované: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")

        y -= 1.0 * cm
        # Metrics block
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(C_TITLE)
        c.drawString(M_LEFT, y, "Súhrnné metriky")
        c.setFillColor(colors.black)
        y -= 0.6 * cm

        rows = [
            ("Svetelné krivky", f"{n_lc:d}"),
            ("Median lc_rms", f"{med_rms:.4f}" if np.isfinite(med_rms) else "—"),
            ("RMS < 0.05 mag", f"{rms_lt_005:d}"),
            ("Avg good comp", f"{avg_good_comp:.2f}" if np.isfinite(avg_good_comp) else "—"),
            ("Best lc_rms", f"{best_rms:.4f}" if np.isfinite(best_rms) else "—"),
            ("Worst lc_rms", f"{worst_rms:.4f}" if np.isfinite(worst_rms) else "—"),
            ("Avg BP-RP", f"{avg_bp_rp:.3f}" if np.isfinite(avg_bp_rp) else "—"),
            ("Setups", f"{setups:d}"),
        ]
        c.setFont("Helvetica", 12)
        for k, v in rows:
            c.drawString(M_LEFT, y, f"{k}:")
            # color key values for rms metrics
            if k in ("Median lc_rms", "Best lc_rms", "Worst lc_rms"):
                c.setFillColor(_metric_color(float(pd.to_numeric(v, errors="coerce"))))
            c.drawString(M_LEFT + 7.5 * cm, y, str(v))
            c.setFillColor(colors.black)
            y -= 0.55 * cm

        y -= 0.4 * cm
        fwhm_txt = f"{fwhm_px:.3f}px" if np.isfinite(fwhm_px) else "—"
        ap_txt = f"{aperture_px:.2f}px" if np.isfinite(aperture_px) else "—"
        c.setFont("Helvetica", 11)
        c.drawString(M_LEFT, y, f"FWHM: {fwhm_txt}   |   Apertura: {ap_txt}")

        # Methods / references
        y -= 1.0 * cm
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(C_TITLE)
        c.drawString(M_LEFT, y, "Metódy")
        c.setFillColor(colors.black)
        y -= 0.6 * cm
        c.setFont("Helvetica", 9)
        c.drawString(M_LEFT, y, "Ensemble fotometria: vážený ZP výpočet w_i = 1/σ_i²")
        y -= 0.45 * cm
        c.setFont("Helvetica", 8)
        c.drawString(
            M_LEFT,
            y,
            "Referencia: Broeg, Ch., Fernández, M., Neuhäuser, R. (2005). "
            "\"A new algorithm for differential photometry: computing an optimum artificial comparison star.\" "
            "Astronomische Nachrichten, 326(2), 134–143. DOI: 10.1002/asna.200410350",
        )

        _page_footer(c)
        c.showPage()

    # ---------------------------------------------------------------------
    # QA page (FWHM + sky + masterstar candidate table)
    # ---------------------------------------------------------------------
    def _draft_id_from_dirname() -> int | None:
        """Best-effort draft_id parse from ``draft_000123`` directory name."""
        nm = str(getattr(draft_dir, "name", "") or "")
        if "draft_" not in nm:
            return None
        try:
            tail = nm.split("draft_", 1)[1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            return int(digits) if digits else None
        except Exception:  # noqa: BLE001
            return None

    def _load_obs_files_for_obs() -> pd.DataFrame:
        """Load QA metrics from DB (OBS_FILES) for this draft + setup."""
        draft_id = _draft_id_from_dirname()
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
            p = str(obs_group).split("_")
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
    def _load_qc_metrics_for_obs() -> pd.DataFrame:
        qc_csv = draft_dir / "processed" / "lights" / "qc_metrics.csv"
        if not qc_csv.exists():
            return pd.DataFrame()
        try:
            dfq = pd.read_csv(qc_csv, low_memory=False)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
        if dfq.empty:
            return pd.DataFrame()
        if "dst" in dfq.columns:
            m = dfq["dst"].astype(str).str.contains(str(obs_group), regex=False)
            dfq = dfq.loc[m].copy()
        if dfq.empty:
            return pd.DataFrame()
        dfq["_dst_name"] = dfq["dst"].astype(str).apply(lambda s: Path(s).name) if "dst" in dfq.columns else ""
        dfq["FWHM_PX"] = pd.to_numeric(dfq.get("fwhm_px"), errors="coerce")
        dfq["SKY_LEVEL"] = pd.to_numeric(dfq.get("bg_median"), errors="coerce")
        dfq["ELONGATION"] = pd.to_numeric(dfq.get("elongation"), errors="coerce")
        if "n_stars_detected" in dfq.columns:
            dfq["STAR_COUNT"] = pd.to_numeric(dfq.get("n_stars_detected"), errors="coerce")
        else:
            dfq["STAR_COUNT"] = pd.to_numeric(dfq.get("n_sources"), errors="coerce")
        dfq = dfq.reset_index(drop=True)
        dfq["frame_index"] = np.arange(1, len(dfq) + 1, dtype=int)
        return dfq

    def _compute_masterstar_score(df: pd.DataFrame) -> pd.Series:
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

    def _infer_masterstar_used_frame(qc_df: pd.DataFrame) -> dict[str, Any] | None:
        ms_fits = platesolve_dir / "MASTERSTAR.fits"
        if not (ms_fits.exists() and qc_df is not None and not qc_df.empty):
            return None
        try:
            from astropy.io import fits

            with fits.open(ms_fits, memmap=False) as hdul:
                hdr = hdul[0].header
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
        if wsum <= 0:
            return None
        i = int(np.nanargmin(d))
        r = qc_df.iloc[i].to_dict()
        r["_match_dist"] = float(d[i])
        return r

    def _draw_qa_page(c: "canvas.Canvas") -> None:
        qc_df = _load_obs_files_for_obs()
        if qc_df.empty:
            qc_df = _load_qc_metrics_for_obs()
        if qc_df.empty:
            return
        try:
            import matplotlib.pyplot as plt
            from io import BytesIO
        except Exception:  # noqa: BLE001
            return

        # Candidates (top5 by score)
        cand = qc_df.copy()
        if "status" in cand.columns:
            cand = cand[cand["status"].astype(str).str.lower().eq("ok")].copy()
        cand["_ms_score"] = _compute_masterstar_score(cand)
        cand = cand.sort_values("_ms_score", ascending=False).reset_index(drop=True)
        top5 = cand.head(5).copy()
        used = _infer_masterstar_used_frame(cand)

        # FWHM limit line (visual): median * 1.05 (match UI auto-default).
        fwhm_vals = pd.to_numeric(qc_df["FWHM_PX"], errors="coerce").to_numpy(dtype=float)
        _med = float(np.nanmedian(fwhm_vals)) if np.isfinite(np.nanmedian(fwhm_vals)) else float("nan")
        fwhm_limit = round(_med * 1.05, 2) if np.isfinite(_med) else float("nan")

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(colors.black)
        c.drawString(M_LEFT, PAGE_H - M_TOP - 0.4 * cm, "FITS Quality Assessment")
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(M_LEFT, PAGE_H - M_TOP - 1.0 * cm, f"Setup: {obs_group}")
        c.setFillColor(colors.black)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.0, 6.2), constrained_layout=True)
        x = qc_df["frame_index"].to_numpy(dtype=int)

        # FWHM plot
        f = pd.to_numeric(qc_df["FWHM_PX"], errors="coerce").to_numpy(dtype=float)
        col = np.where(np.isfinite(f) & np.isfinite(fwhm_limit) & (f <= fwhm_limit), "#27ae60", "#c0392b")
        ax1.plot(x, f, color="#6c5ce7", linewidth=1.1, alpha=0.65)
        ax1.scatter(x, f, c=col, s=20)
        if np.isfinite(fwhm_limit):
            ax1.axhline(y=float(fwhm_limit), color="#c0392b", linestyle="--", linewidth=1.3)
        ax1.set_xlabel("Frame Index")
        ax1.set_ylabel("FWHM (px)")
        ax1.set_title("FWHM — zelená ≤ limit, červená > limit")
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
        ax2.set_title("Background sky level (podľa filtra; červená = auto-outlier)")
        ax2.grid(True, alpha=0.25)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        img = ImageReader(buf)
        plots_h = 11.0 * cm
        c.drawImage(
            img,
            M_LEFT,
            PAGE_H - M_TOP - 1.6 * cm - plots_h,
            width=USE_W,
            height=plots_h,
            preserveAspectRatio=True,
            mask="auto",
        )

        # Table section under plots
        y0 = PAGE_H - M_TOP - 1.6 * cm - plots_h - 0.6 * cm
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(colors.black)
        c.drawString(M_LEFT, y0, "Masterstar referenčný snímok")
        y0 -= 0.45 * cm
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#333333"))
        if used:
            used_fn = str(used.get("_dst_name") or "")
            used_fr = int(pd.to_numeric(used.get("frame_index"), errors="coerce") or 0)
            used_fwhm = pd.to_numeric(used.get("FWHM_PX"), errors="coerce")
            used_sc = pd.to_numeric(used.get("_ms_score"), errors="coerce")
            if np.isfinite(used_fwhm) and np.isfinite(used_sc):
                c.drawString(
                    M_LEFT,
                    y0,
                    f"Použitý snímok: {used_fn}  (Frame {used_fr}, FWHM={float(used_fwhm):.2f}, Score={float(used_sc):.3f})",
                )
            else:
                c.drawString(M_LEFT, y0, f"Použitý snímok: {used_fn}  (Frame {used_fr})")
        else:
            c.drawString(M_LEFT, y0, "Použitý snímok: — (nedá sa inferovať z dostupných metadát)")
        c.setFillColor(colors.black)
        y0 -= 0.7 * cm

        c.setFont("Helvetica-Bold", 11)
        c.drawString(M_LEFT, y0, "Top 5 kandidátov pre Masterstar")
        y0 -= 0.4 * cm

        headers = ["Poradie", "Frame", "Filename", "FWHM", "Elongácia", "Hviezdy", "Sky", "Skóre"]
        rows: list[list[str]] = [headers]
        for i in range(len(top5)):
            r = top5.iloc[i]
            rows.append(
                [
                    str(i + 1),
                    str(int(pd.to_numeric(r.get("frame_index"), errors="coerce") or 0)),
                    str(r.get("_dst_name") or ""),
                    f"{float(pd.to_numeric(r.get('FWHM_PX'), errors='coerce')):.2f}"
                    if np.isfinite(pd.to_numeric(r.get("FWHM_PX"), errors="coerce"))
                    else "—",
                    f"{float(pd.to_numeric(r.get('ELONGATION'), errors='coerce')):.3f}"
                    if np.isfinite(pd.to_numeric(r.get("ELONGATION"), errors="coerce"))
                    else "—",
                    f"{int(pd.to_numeric(r.get('STAR_COUNT'), errors='coerce') or 0)}"
                    if np.isfinite(pd.to_numeric(r.get("STAR_COUNT"), errors="coerce"))
                    else "—",
                    f"{float(pd.to_numeric(r.get('SKY_LEVEL'), errors='coerce')):.0f}"
                    if np.isfinite(pd.to_numeric(r.get("SKY_LEVEL"), errors="coerce"))
                    else "—",
                    f"{float(pd.to_numeric(r.get('_ms_score'), errors='coerce')):.3f}"
                    if np.isfinite(pd.to_numeric(r.get("_ms_score"), errors="coerce"))
                    else "—",
                ]
            )

        t = Table(
            rows,
            colWidths=[1.6 * cm, 1.6 * cm, 6.2 * cm, 1.8 * cm, 2.2 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm],
            rowHeights=[0.55 * cm] * len(rows),
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), C_TITLE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ]
            )
        )
        tw, th = t.wrap(USE_W, y0 - M_BOTTOM)
        t.drawOn(c, M_LEFT, y0 - th)

        _page_footer(c)
        c.showPage()

    def _comp_rows_for_target(cid: str) -> list[list[str]]:
        if comp_df.empty or not cid:
            return []
        if "_tcid" not in comp_df.columns:
            return []
        sub = comp_df[comp_df["_tcid"] == cid].copy()
        if sub.empty:
            return []
        # Normalize dist column name
        if "_dist_deg" in sub.columns and "dist_deg" not in sub.columns:
            sub["dist_deg"] = sub["_dist_deg"]

        # Optional per-target comp quality from Phase2A (saved as JSON).
        qmap: dict[str, str] = {}
        try:
            qpath = lc_dir / f"comp_quality_{cid}.json"
            if qpath.exists():
                import json

                qraw = json.loads(qpath.read_text(encoding="utf-8"))
                if isinstance(qraw, dict):
                    qmap = {str(k): str(v) for k, v in qraw.items()}
        except Exception:  # noqa: BLE001
            qmap = {}

        def _fmt(v: Any, nd: int) -> str:
            x = pd.to_numeric(v, errors="coerce")
            return f"{float(x):.{nd}f}" if np.isfinite(x) else "—"

        # Relative weights within target (normalize to max=1.0); ignore NaN/inf/non-positive.
        max_w = float("nan")
        try:
            if "comp_weight" in sub.columns:
                wv = pd.to_numeric(sub["comp_weight"], errors="coerce")
                wv = wv[np.isfinite(wv.to_numpy(dtype=float)) & (wv.to_numpy(dtype=float) > 0)]
                max_w = float(wv.max()) if wv.size else float("nan")
        except Exception:  # noqa: BLE001
            max_w = float("nan")

        out: list[list[str]] = []
        sub = sub.reset_index(drop=True)
        for i in range(len(sub)):
            r = sub.iloc[i]
            # 'stav' may be stored under different column names; fallback to comp_quality json.
            stav = ""
            for col in ("stav", "quality", "comp_quality"):
                if col in sub.columns:
                    stav = str(r.get(col, "") or "").strip()
                    break
            if not stav:
                try:
                    ccid = _norm_cid(r.get("catalog_id", ""))
                    stav = str(qmap.get(ccid, "") or "").strip()
                except Exception:  # noqa: BLE001
                    stav = ""
            out.append(
                [
                    str(i + 1),
                    _fmt(r.get("mag"), 3),
                    _fmt(r.get("b_v"), 3),
                    _fmt(r.get("bp_rp"), 3),
                    _fmt(r.get("dist_deg"), 4),
                    str(int(pd.to_numeric(r.get("comp_n_frames"), errors="coerce"))) if np.isfinite(pd.to_numeric(r.get("comp_n_frames"), errors="coerce")) else "—",
                    _fmt(r.get("comp_rms"), 4),
                    _fmt((float(pd.to_numeric(r.get("comp_weight"), errors="coerce")) / max_w) if np.isfinite(pd.to_numeric(r.get("comp_weight"), errors="coerce")) and float(pd.to_numeric(r.get("comp_weight"), errors="coerce")) > 0 and np.isfinite(max_w) and max_w > 0 else float("nan"), 3),
                    str(r.get("comp_tier", "") or ""),
                    stav,
                ]
            )
        return out

    def draw_star_page(c: "canvas.Canvas", star_data: dict[str, Any]) -> None:
        c.setPageSize(landscape(A4))

        y_cursor = PAGE_H - M_TOP

        vsx_name = str(star_data.get("vsx_name", "") or "")
        vsx_type = str(star_data.get("vsx_type", "") or "")
        zone_flag = str(star_data.get("zone_flag", "") or "")
        bp_rp_val = star_data.get("bp_rp", float("nan"))
        try:
            bp_rp_f = float(bp_rp_val)
        except Exception:
            bp_rp_f = float("nan")
        bp_rp_txt = f"{bp_rp_f:.3f}" if np.isfinite(bp_rp_f) else "—"

        title = f"{vsx_name}  |  {vsx_type}  |  {zone_flag}  |  BP-RP: {bp_rp_txt}"
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.black)
        c.drawString(M_LEFT, y_cursor - 0.6 * cm, title)
        y_cursor -= TITLE_H

        # Metrics
        lc_rms = float(pd.to_numeric(star_data.get("lc_rms"), errors="coerce"))
        good_comp = int(pd.to_numeric(star_data.get("good_comp"), errors="coerce") or 0)
        ap_px = float(pd.to_numeric(star_data.get("aperture_px"), errors="coerce"))
        rms_txt = f"{lc_rms:.4f}" if np.isfinite(lc_rms) else "—"
        ap_txt = f"{ap_px:.1f}px" if np.isfinite(ap_px) else "—"
        metrics = f"lc_rms: {rms_txt}   |   good comp: {good_comp:d}   |   apertura: {ap_txt}   |   Vizier   VSX"
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(M_LEFT, y_cursor - 0.35 * cm, metrics)
        c.setFillColor(colors.black)
        y_cursor -= METRICS_H

        # Separator
        c.setStrokeColor(colors.HexColor("#cccccc"))
        c.setLineWidth(0.8)
        c.line(M_LEFT, y_cursor, PAGE_W - M_RIGHT, y_cursor)
        y_cursor -= (SEP_H + 0.1 * cm)

        # -----------------------------------------------------------------
        # Dynamic graph height so COMP table never overflows the page.
        # -----------------------------------------------------------------
        comp_rows = star_data.get("comp_rows") or []
        n_comp_rows = int(len(comp_rows))
        ROW_H = 0.55 * cm
        HEADER_H = 0.60 * cm
        TABLE_MARGIN = 0.5 * cm  # space above table
        NOTE_H = 0.4 * cm  # note + footer reserve

        table_h_needed = HEADER_H + n_comp_rows * ROW_H + NOTE_H + TABLE_MARGIN
        FIXED_TOP = 1.5 * cm  # title + metrics + separator (approx)
        # Graph height available between top block and table block.
        graph_h = PAGE_H - M_TOP - FIXED_TOP - table_h_needed - M_BOTTOM
        graph_h = max(graph_h, 6.0 * cm)
        graph_h = min(graph_h, 15.0 * cm)

        # Graphic section: lightcurve + field image
        lc_x = M_LEFT
        fi_x = M_LEFT + LC_W + GAP_W
        y_top = y_cursor

        lc_img = star_data.get("lc_img")
        fi_img = star_data.get("field_img")
        if lc_img and Path(lc_img).exists():
            _draw_image_fit(c, Path(lc_img), lc_x, y_top, LC_W, graph_h)
        else:
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#333333"))
            c.drawString(lc_x + 1.0 * cm, y_top - graph_h / 2.0, "Svetelná krivka nie je k dispozícii")
            c.setFillColor(colors.black)

        if fi_img and Path(fi_img).exists():
            _draw_image_fit(c, Path(fi_img), fi_x, y_top, FI_W, graph_h)

        # Place the table immediately under the graph box (no extra whitespace).
        y_cursor = y_top - graph_h

        # Separator
        c.setStrokeColor(colors.HexColor("#cccccc"))
        c.setLineWidth(0.8)
        c.line(M_LEFT, y_cursor, PAGE_W - M_RIGHT, y_cursor)
        y_cursor -= (SEP_H + 0.05 * cm)

        # COMP table
        if comp_rows:
            headers = ["#", "mag", "B-V", "BP-RP", "dist_deg", "n_frames", "p2p RMS", "w (rel)", "tier", "stav"]
            table_data = [headers] + comp_rows
            col_widths = [1.2 * cm, 1.8 * cm, 1.6 * cm, 1.8 * cm, 2.2 * cm, 2.0 * cm, 2.2 * cm, 1.8 * cm, 2.0 * cm, 2.0 * cm]
            row_heights = [0.55 * cm] * len(table_data)

            t = Table(table_data, colWidths=col_widths, rowHeights=row_heights)
            style = TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), C_TITLE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                ]
            )
            # Row background based on stav (last col)
            for i in range(1, len(table_data)):
                stav = str(table_data[i][-1] or "").strip().lower()
                if stav == "good":
                    bg = colors.HexColor("#d4edda")
                elif stav == "suspect":
                    bg = colors.HexColor("#fff3cd")
                else:
                    bg = colors.HexColor("#f8d7da") if stav else colors.white
                style.add("BACKGROUND", (0, i), (-1, i), bg)
            t.setStyle(style)

            t_w, t_h = t.wrap(USE_W, y_cursor - M_BOTTOM)
            table_y = y_cursor - TABLE_MARGIN
            t.drawOn(c, M_LEFT, table_y - t_h)
            y_cursor = table_y - t_h - 0.1 * cm
        else:
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#333333"))
            c.drawString(M_LEFT, y_cursor - 0.4 * cm, "COMP tabuľka nie je k dispozícii")
            c.setFillColor(colors.black)

        # Note
        c.setFont("Helvetica-Oblique", 6)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(M_LEFT, M_BOTTOM + 0.2 * cm, NOTE_TXT)
        c.setFillColor(colors.black)

        _page_footer(c)
        c.showPage()

    def _draw_summary_page(c: "canvas.Canvas") -> None:
        # Compact table of all stars sorted by lc_rms
        y = PAGE_H - M_TOP
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(C_TITLE)
        c.drawString(M_LEFT, y - 0.5 * cm, "Súhrn všetkých hviezd (zoradené podľa lc_rms)")
        c.setFillColor(colors.black)
        y -= 1.0 * cm

        cols = [cname for cname in ("vsx_name", "vsx_type", "lc_rms", "n_good_comp", "zone_flag", "bp_rp") if cname in summary_df.columns]
        if not cols:
            _page_footer(c)
            c.showPage()
            return
        work = summary_df[cols].copy()
        work["lc_rms"] = pd.to_numeric(work.get("lc_rms"), errors="coerce")
        work["bp_rp"] = pd.to_numeric(work.get("bp_rp"), errors="coerce")

        data: list[list[str]] = [cols]
        for _, r in work.iterrows():
            row_out: list[str] = []
            for cn in cols:
                if cn == "lc_rms":
                    x = r.get(cn)
                    row_out.append(f"{float(x):.4f}" if np.isfinite(x) else "—")
                elif cn == "bp_rp":
                    x = r.get(cn)
                    row_out.append(f"{float(x):.3f}" if np.isfinite(x) else "—")
                else:
                    row_out.append(str(r.get(cn, "") or ""))
            data.append(row_out)

        col_widths = [6.0 * cm, 2.0 * cm, 2.0 * cm, 2.4 * cm, 3.0 * cm, 2.0 * cm]
        col_widths = col_widths[: len(cols)]
        table_style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), C_TITLE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )

        # Split summary table into multiple pages to avoid overflowing the page height.
        ROW_H = 0.55 * cm
        HEADER_H = 0.60 * cm
        table_bottom_y = M_BOTTOM + 1.0 * cm
        avail_h = (y - table_bottom_y)
        rows_per_page = int(max(15, min(60, (avail_h - HEADER_H) // ROW_H)))

        header_row = data[0]
        data_rows = data[1:]
        for chunk_start in range(0, len(data_rows), rows_per_page):
            chunk = [header_row] + data_rows[chunk_start : chunk_start + rows_per_page]
            t = Table(chunk, colWidths=col_widths, rowHeights=[ROW_H] * len(chunk))
            t.setStyle(table_style)
            t.wrapOn(c, USE_W, USE_H)
            t.drawOn(c, M_LEFT, table_bottom_y)
            _page_footer(c)
            if chunk_start + rows_per_page < len(data_rows):
                c.showPage()
                # Re-draw the title on continuation pages.
                y2 = PAGE_H - M_TOP
                c.setFont("Helvetica-Bold", 14)
                c.setFillColor(C_TITLE)
                c.drawString(M_LEFT, y2 - 0.5 * cm, "Súhrn všetkých hviezd (zoradené podľa lc_rms) — pokračovanie")
                c.setFillColor(colors.black)
                y = y2 - 1.0 * cm
        c.showPage()

    # Build PDF
    c = canvas.Canvas(str(output_pdf), pagesize=landscape(A4))

    # 1) Cover page
    _draw_cover_page(c)

    # 2) QA page (optional; only if qc_metrics.csv exists)
    _draw_qa_page(c)

    # 3) Stars pages
    for _, row in summary_df.iterrows():
        try:
            cid = str(row.get("_cid", "") or _norm_cid(row.get("catalog_id", "")))
            vsx_name = str(row.get("vsx_name", cid) or cid)

            # Lightcurve image (prefer existing PNG -> jpeg; else generate from CSV)
            lc_png = Path(str(row.get("lc_png", "") or "")).expanduser()
            if not lc_png.is_absolute():
                lc_png = (lc_dir / lc_png.name) if lc_png.name else lc_png
            lc_csv = Path(str(row.get("lc_csv", "") or "")).expanduser()
            if not lc_csv.is_absolute():
                lc_csv = (lc_dir / lc_csv.name) if lc_csv.name else lc_csv

            lc_img: Path | None = None
            if lc_png.exists():
                lc_img = _prepare_jpeg(lc_png, cache_dir / f"lc_{cid}.jpg", max_side_px=1800, quality=75)
            else:
                lc_img = _plot_lightcurve_to_jpeg(lc_csv, cache_dir / f"lc_{cid}.jpg")

            # Field image (field_map per target preferred)
            field_img = None
            fm = lc_dir / f"field_map_{cid}.png"
            if fm.exists():
                field_img = fm
            else:
                for cand in (
                    platesolve_dir / "masterstar_field.png",
                    photometry_dir / f"field_{cid}.png",
                    photometry_dir / f"field_{vsx_name}.png",
                ):
                    if cand.exists():
                        field_img = cand
                        break
            field_img_jpg = _prepare_jpeg(Path(field_img), cache_dir / f"field_{cid}.jpg", max_side_px=1400, quality=70) if field_img else None

            star_data = {
                "catalog_id": cid,
                "vsx_name": vsx_name,
                "vsx_type": row.get("vsx_type", ""),
                "zone_flag": row.get("zone_flag", ""),
                "bp_rp": row.get("bp_rp", float("nan")),
                "lc_rms": row.get("lc_rms", float("nan")),
                "good_comp": row.get("n_good_comp", 0),
                "aperture_px": row.get("aperture_px", float("nan")),
                "lc_img": str(lc_img) if lc_img is not None else "",
                "field_img": str(field_img_jpg) if field_img_jpg is not None else "",
                "comp_rows": _comp_rows_for_target(cid),
            }
            draw_star_page(c, star_data)
        except Exception as exc_star:  # noqa: BLE001
            logging.warning("PDF: preskakujem hviezdu (%s): %s", row.get("vsx_name", ""), exc_star)
            continue

    # 4) Summary page
    _draw_summary_page(c)

    c.save()
    return output_pdf

