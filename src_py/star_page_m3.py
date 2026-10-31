"""STAR-PAGE-M3-2026-10: content of the per-star PDF page (layout approved by Milan 2026-10-11).

Pure helpers (no reportlab): target naming, the light-curve + check-star figure with a UT
top axis, the aperture / sky-annulus cutout, the facts list and the comparison-star rows.
The page itself is drawn by ``photometry_report._PhotometryReportBuilder._report_per_star_page``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gaia_catalog_id import read_vyvar_csv

from mag_constants import MAG_ERR_SCALE


def _s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and not math.isfinite(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def _f(v: Any) -> float:
    try:
        x = float(pd.to_numeric(v, errors="coerce"))
    except (TypeError, ValueError):
        return float("nan")
    return x


def target_display_name(row: Mapping[str, Any] | None, cid: str = "") -> str:
    """Name of the measured object: VSX name, else exoplanet host (TOI-xxxx, HAT-P-x ...),
    else the target's own name, else ``Gaia DR3 <id>`` (Milan 2026-10-11: a TOI that is in
    neither Gaia variables nor VSX must still be named as TOI)."""
    row = {} if row is None else row
    cid = _s(cid) or _s(row.get("catalog_id"))
    for k in ("vsx_name", "exo_host_name", "name"):
        v = _s(row.get(k))
        if v and v != cid and not v.isdigit():
            return v
    return f"Gaia DR3 {cid}" if cid else "unknown"


def target_source_line(row: Mapping[str, Any] | None) -> str:
    """Subtitle by catalogue source: VSX type + period / exoplanet source + disposition."""
    row = {} if row is None else row
    parts: list[str] = []
    vt = _s(row.get("vsx_type"))
    if _s(row.get("vsx_name")):
        parts.append(f"VSX {vt}" if vt else "VSX")
        p = _f(row.get("vsx_period"))
        if math.isfinite(p) and p > 0:
            parts.append(f"P = {p:.6f} d")
    if _s(row.get("exo_host_name")) or _s(row.get("target_origin")).upper() == "EXOPLANET":
        src = _s(row.get("exo_cat_source"))
        disp = _s(row.get("exo_disposition"))
        parts.append("exoplanet host" + (f" ({src}{', ' + disp if disp else ''})" if src or disp else ""))
    return "  |  ".join(parts)


def ra_dec_text(ra: float, dec: float) -> str:
    """'hh:mm:ss.ss +dd:mm:ss.s'; rounding carries into minutes / hours (G11-37: never ':60.00')."""
    if not (math.isfinite(ra) and math.isfinite(dec)):
        return "-"
    from astropy import units as u  # noqa: PLC0415
    from astropy.coordinates import Angle  # noqa: PLC0415

    ra_t = Angle(ra % 360.0, unit=u.deg).to_string(unit=u.hourangle, sep=":", precision=2, pad=True)
    de_t = Angle(dec, unit=u.deg).to_string(unit=u.deg, sep=":", precision=1, pad=True, alwayssign=True)
    return f"{ra_t} {de_t}"


def _jd_to_hhmm(jd: float) -> str:
    if not math.isfinite(jd):
        return ""
    frac = (jd + 0.5) % 1.0
    m = int(round(frac * 1440.0)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def lc_plot_data(df: pd.DataFrame, chk: pd.DataFrame | None, *, uncalibrated: bool) -> dict[str, Any] | None:
    """G14-17: the numbers of the light-curve figure, shared by the PDF (matplotlib) and the UI (plotly).

    Points: exported = ``photometry_lightcurve.export_row_mask`` (the rows of the AAVSO / VarAstro files),
    the other plotted points are not exported. Check strip: ``check_star_kmag.check_star_stats`` on the
    exported epochs (the numbers of the export header and the trust), around its median. ``ut_offset``:
    JD(UTC) = x + ut_offset on the plotted axis. None when fewer than 2 points can be drawn."""
    from check_star_kmag import check_star_stats  # noqa: PLC0415
    from photometry_lightcurve import export_row_mask, published_mag_column  # noqa: PLC0415

    if df is None or df.empty:
        return None
    ycol = "delta_mag" if uncalibrated else (published_mag_column(df) or "mag_calib")
    if ycol not in df.columns:
        return None
    xcol = "bjd" if "bjd" in df.columns else "jd"
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
    e = MAG_ERR_SCALE * pd.to_numeric(df["err"], errors="coerce").to_numpy(float) if "err" in df.columns else None
    fl = (df["flag"].astype(str).str.strip().str.lower().to_numpy() if "flag" in df.columns
          else np.array(["normal"] * len(df)))
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 2:
        return None
    exp = export_row_mask(df, mag_column=ycol).to_numpy(bool)
    good = ok & exp
    bad = ok & ~exp
    from jd_axis_format import jd_offset  # noqa: PLC0415

    x0 = float(jd_offset(x[ok]) or 0)  # G11-29: no offset for times that are not full JD
    ut_offset = float("nan")
    if "jd" in df.columns:
        jd = pd.to_numeric(df["jd"], errors="coerce").to_numpy(float)
        okj = ok & np.isfinite(jd)
        if int(okj.sum()) >= 2:
            # JD - (BJD - x0) changes by < 1 s over a night: a constant offset is exact enough
            ut_offset = float(np.median(jd[okj] - (x[okj] - x0)))
    st = check_star_stats(chk, df.loc[exp, "source_file"] if "source_file" in df.columns else [])
    has_check = st.n >= 2 and bool(np.isfinite(st.bjd).all())
    kx = st.bjd - x0 if has_check else np.array([])
    kr = st.kmag - st.median if has_check else np.array([])
    lim = max(0.03, float(np.percentile(np.abs(kr), 98)) * 1.3) if has_check else 0.03
    return {
        "xcol": xcol, "ycol": ycol, "x": x, "y": y, "e": e, "flag": fl, "ok": ok, "good": good, "bad": bad,
        "x0": x0, "ut_offset": ut_offset, "has_check": has_check, "kx": kx, "kr": kr, "check_lim": lim,
        "check_text": f"check star scatter {st.scatter:.3f} mag" if has_check else "",
        "xlabel": f"BJD_TDB - {x0:.0f}" if xcol == "bjd" else f"JD - {x0:.0f}",
    }


def plot_lc_check(
    lc_csv: Path,
    out_png: Path,
    *,
    check_csv: Path | None = None,
    uncalibrated: bool = False,
    gspc_band: str = "",
    show_gspc: bool = False,
    figsize: tuple[float, float] = (10.0, 5.6),
) -> Path | None:
    """Light curve (exported points blue, flagged points red, error bars in mag) with a
    check-star strip on the same time axis and UT (HH:MM) on the top axis. Uncalibrated:
    delta_mag on the left axis; with ``show_gspc`` a second right scale g GSPC (same
    quantity, shifted - one axis, relabelled)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(lc_csv, low_memory=False)
    chk = None
    if check_csv is not None and Path(check_csv).is_file():
        try:
            chk = read_vyvar_csv(check_csv, low_memory=False)
        except (OSError, ValueError):
            chk = None
    pd_ = lc_plot_data(df, chk, uncalibrated=uncalibrated)
    if pd_ is None:
        return None
    x, y, e, ok, x0 = pd_["x"], pd_["y"], pd_["e"], pd_["ok"], pd_["x0"]
    exp_mask, bad_mask, ycol = pd_["good"], pd_["bad"], pd_["ycol"]
    has_check = pd_["has_check"]

    fig = plt.figure(figsize=figsize, dpi=150)
    if has_check:
        gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.0], hspace=0.06)
        ax = fig.add_subplot(gs[0])
        axk = fig.add_subplot(gs[1], sharex=ax)
    else:
        ax = fig.add_subplot(111)
        axk = None
    try:
        if e is not None:
            ax.errorbar(x[exp_mask] - x0, y[exp_mask], yerr=e[exp_mask], fmt="o", ms=3.2, color="#0b5cad",
                        ecolor="#9cc3e8", elinewidth=0.8, capsize=0, label="exported", zorder=3)
        else:
            ax.plot(x[exp_mask] - x0, y[exp_mask], "o", ms=3.2, color="#0b5cad", label="exported")
        if bad_mask.any():
            ax.plot(x[bad_mask] - x0, y[bad_mask], "x", ms=4.5, color="#c0392b", label="flagged (not exported)", zorder=4)
            ax.legend(fontsize=7, loc="best", frameon=False)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("delta_mag (not calibrated)" if uncalibrated
                      else (f"{gspc_band} [mag]" if gspc_band else f"{ycol}"), fontsize=9)
        if show_gspc and "mag_gspc" in df.columns:
            yg = pd.to_numeric(df["mag_gspc"], errors="coerce").to_numpy(float)
            okg = ok & np.isfinite(yg)
            if int(okg.sum()) >= 2:
                off = float(np.median(yg[okg] - y[okg]))
                a2 = ax.secondary_yaxis("right", functions=(lambda v, o=off: v + o, lambda v, o=off: v - o))
                a2.set_ylabel(f"{gspc_band} GSPC (same scale, shifted)", fontsize=8, color="#57606a")
                a2.tick_params(labelsize=7, colors="#57606a")
        # UT (HH:MM) on the top axis: JD(UTC) mid-exposure on the plotted time (lc_plot_data ut_offset)
        _dj = pd_["ut_offset"]
        if math.isfinite(_dj):
            at = ax.secondary_xaxis("top", functions=(lambda v, d=_dj: v + d, lambda v, d=_dj: v - d))
            at.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0 / 24.0))  # full UT hours
            at.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: _jd_to_hhmm(v)))
            at.tick_params(labelsize=7)
            at.set_xlabel("UT (HH:MM)", fontsize=8)
        if axk is not None:
            axk.plot(pd_["kx"], pd_["kr"], "o", ms=2.4, color="#6e7781")
            axk.axhline(0, color="#d0d7de", lw=0.8)
            axk.invert_yaxis()
            lim = pd_["check_lim"]
            axk.set_ylim(lim, -lim)
            axk.grid(True, alpha=0.3, lw=0.5)
            axk.tick_params(labelsize=7)
            axk.set_ylabel("check - med", fontsize=8)
            axk.text(0.01, 0.92, pd_["check_text"], transform=axk.transAxes,
                     fontsize=7, color="#57606a", va="top")
            plt.setp(ax.get_xticklabels(), visible=False)
            axk.set_xlabel(pd_["xlabel"], fontsize=9)
        else:
            ax.set_xlabel(pd_["xlabel"], fontsize=9)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out_png


def render_cutout(
    image_fits: Path | np.ndarray,
    x: float,
    y: float,
    r_ap: float,
    r_in: float,
    r_out: float,
    out_png: Path,
    *,
    neighbours: list[tuple[float, float, float]] | None = None,
    half: int | None = None,
) -> Path | None:
    """Target cutout with the aperture (blue) and the sky annulus (orange); stars brighter
    than G 17 inside the annulus are labelled (they bias the sky estimate). ``image_fits`` is the
    MASTERSTAR path or its already loaded data (the report renders every target from one read)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from matplotlib.patches import Circle

    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r_out)):
        return None
    data = image_fits if isinstance(image_fits, np.ndarray) else fits.getdata(image_fits)
    if data is None:
        return None
    data = np.asarray(data, dtype=float)
    hw = int(half or math.ceil(r_out * 1.6))
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi - hw), min(data.shape[0], yi + hw + 1)
    x0, x1 = max(0, xi - hw), min(data.shape[1], xi + hw + 1)
    cut = data[y0:y1, x0:x1]
    if cut.size == 0:
        return None
    lo, hi = np.nanpercentile(cut, [5, 99.7])
    fig = plt.figure(figsize=(3.2, 3.2), dpi=150)
    try:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(np.arcsinh(np.clip((cut - lo) / max(hi - lo, 1e-9), -0.1, None) * 10), cmap="gray_r", origin="lower",
                  extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5))
        ax.add_patch(Circle((x, y), r_ap, fill=False, lw=1.4, color="#0b5cad"))
        for r in (r_in, r_out):
            ax.add_patch(Circle((x, y), r, fill=False, lw=1.2, ls="--", color="#bc4c00"))
        # BLEND-CUTOUT-2026-10: every Gaia neighbour brighter than G 17 inside the sky annulus's
        # outer radius - in the aperture (blend light, red) or between aperture and annulus / in the
        # annulus (sky, orange) - is marked, also when it is not resolved on the image
        for nx, ny, g in neighbours or []:
            d = math.hypot(nx - x, ny - y)
            if d < r_out and math.isfinite(g) and g < 17.0:
                col = "#c62828" if d <= r_ap else "#bc4c00"
                ax.plot([nx], [ny], marker="+", ms=9, mew=1.4, color=col)
                ax.annotate(f"G {g:.1f}", (nx, ny), xytext=(4, 4), textcoords="offset points", fontsize=8, color=col)
        ax.set_xlim(x0 - 0.5, x1 - 0.5)
        ax.set_ylim(y0 - 0.5, y1 - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
    finally:
        plt.close(fig)
    return out_png


def load_comp_quality(lc_dir: Path, cid: str) -> dict[str, Any]:
    try:
        raw = json.loads((Path(lc_dir) / f"comp_quality_{cid}.json").read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def short_status(entry: Any) -> str:
    """'suspect* 5.1 mmag/h' instead of the multi-line aperture-photometry note (footnote explains *)."""
    if not isinstance(entry, dict):
        return str(entry or "")
    q = _s(entry.get("quality")) or "good"
    note = _s(entry.get("note"))
    if q == "suspect":
        m = re.search(r"slope=([\d.]+)\s*mmag/hr", note)
        return f"suspect* {m.group(1)} mmag/h" if m else "suspect*"
    return q


def gspc_cell(cid: str, gz: Mapping[str, Any], gspc_rows: Mapping[str, Mapping[str, Any]], band: str) -> str:
    """Accepted GSPC magnitude, or '(m) f0' / '(m) bl' / '(m) S/N' for rejected comps."""
    from gspc_zp import GSPC_BAND, gspc_mag  # noqa: PLC0415

    row = gspc_rows.get(cid)
    p = GSPC_BAND.get(band)
    if row is None or p is None:
        return "-"
    m_raw = _f(row.get(f"{p}_mag"))
    if not math.isfinite(m_raw):
        return "-"
    if cid in set(gz.get("accepted") or []):
        return f"{m_raw:.3f}"
    why = gspc_mag(row, band)[2]
    tag = {"flag 0 (extrapolation)": "f0", "XP spectrum blended": "bl", "S/N <= 30": "S/N", "C* outlier": "C*"}.get(why, "x")
    return f"({m_raw:.2f}) {tag}"


def comp_rows(
    comp_sub: pd.DataFrame,
    cq: Mapping[str, Any],
    *,
    band: str,
    gspc_rows: Mapping[str, Mapping[str, Any]] | None = None,
    target_bp_rp: float = float("nan"),
    max_rows: int = 12,
) -> tuple[list[list[str]], list[str], int]:
    """Rows: #, Gaia DR3, G, BP-RP, dBP-RP, dist ('), rms (mmag), w (Broeg, sum 1), g GSPC, status.
    Excluded comps go to the returned note list; the int is the number of rows not shown."""
    # G11-38: w as written to the AAVSO NOTES / VarAstro header - the Broeg weights of the ensemble, sum 1
    w = cq.get("broeg_weights") if isinstance(cq.get("broeg_weights"), dict) else {}
    gz = cq.get("gspc_zero_point") if isinstance(cq.get("gspc_zero_point"), dict) else {}
    rows: list[list[str]] = []
    excluded: list[str] = []
    for _, r in comp_sub.iterrows():
        cid = _s(r.get("_ccid")) or _s(r.get("catalog_id"))
        ent = cq.get(cid, {})
        if isinstance(ent, dict) and _s(ent.get("quality")) == "excluded":
            excluded.append(f"{cid[-6:]} ({_s(ent.get('note')) or 'excluded'})")
            continue
        bp = _f(r.get("bp_rp"))
        dbp = abs(bp - target_bp_rp) if math.isfinite(bp) and math.isfinite(target_bp_rp) else _f(r.get("delta_bprp_abs"))
        dist = _f(r.get("_dist_deg", r.get("dist_deg")))
        rms = _f(r.get("comp_rms"))
        wr = _f(w.get(cid))
        rows.append([
            str(len(rows) + 1), cid, f"{_f(r.get('mag')):.2f}", f"{bp:.2f}" if math.isfinite(bp) else "-",
            f"{dbp:.2f}" if math.isfinite(dbp) else "-", f"{dist * 60:.1f}" if math.isfinite(dist) else "-",
            f"{rms * 1000.0:.1f}" if math.isfinite(rms) else "-", f"{wr:.3f}" if math.isfinite(wr) else "-",
            gspc_cell(cid, gz, gspc_rows or {}, band) if gspc_rows else "-",
            short_status(ent) + (" (reserve)" if _s(r.get("used_role") or r.get("comp_role")) == "replacement" else ""),
        ])
    hidden = max(0, len(rows) - max_rows)
    return rows[:max_rows], excluded, hidden


COMP_HEADERS = ["#", "Gaia DR3", "G", "BP-RP", "dBP-RP", "dist '", "rms mmag", "w", "{band} GSPC", "status"]
COMP_FOOT = ("* suspect = aperture-photometry stability slope; kept only when fewer than 3 clean comps exist (COMP-SMART); "
             "(reserve) = replaced a suspect comp.  GSPC: accepted value; "
             "(m) f0 = flag 0 (extrapolation), bl = XP spectrum blended, S/N = below 30 - not used.  "
             "w = Broeg (2005) weight in the ensemble (weights sum to 1, as in the AAVSO NOTES).  "
             "rms = comparison-star scatter (leave-one-out, mmag).")


def n_eff(weights: Mapping[str, float]) -> float:
    v = np.array([float(x) for x in weights.values() if math.isfinite(float(x)) and float(x) > 0])
    return float(v.sum() ** 2 / (v ** 2).sum()) if v.size else float("nan")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column, all-NaN when absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _aperture_choice(lc_csv: Path) -> tuple[float, float]:
    """(night FWHM px, k = f_star) of the target of ``lightcurve_<cid>.csv`` from
    ``photometry/aperture_per_target.json``; NaN when the file or the target is missing."""
    cid = lc_csv.stem[len("lightcurve_"):] if lc_csv.stem.startswith("lightcurve_") else ""
    try:
        apt = json.loads((lc_csv.parent.parent / "aperture_per_target.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return float("nan"), float("nan")
    if not isinstance(apt, dict):
        return float("nan"), float("nan")

    def _f(v: Any) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return x if math.isfinite(x) else float("nan")

    ent = (apt.get("targets") or {}).get(cid) or {}
    return _f(apt.get("fwhm_night_px")), _f(ent.get("f_star"))


def lc_facts(lc_csv: Path, *, mag_column: str | None = None) -> dict[str, Any]:
    """Frames exported / not exported (``export_row_mask`` with the target's export magnitude column, G11-12),
    span, airmass range, aperture radius (median, k), FWHM night."""
    out: dict[str, Any] = {}
    try:
        df = pd.read_csv(lc_csv, low_memory=False)
    except (OSError, ValueError):
        return out
    from photometry_lightcurve import export_row_mask, lc_export_mag_column  # noqa: PLC0415

    fl = df["flag"].astype(str).str.strip().str.lower() if "flag" in df.columns else pd.Series(["normal"] * len(df))
    exp = export_row_mask(df, mag_column=mag_column or lc_export_mag_column(df))
    out["n_total"] = int(len(df))
    out["n_exported"] = int(exp.sum())
    out["n_excluded"] = out["n_total"] - out["n_exported"]
    # exported with FLAG=<flag> in the AAVSO NOTES (REPORT-AUDIT-2026-10)
    out["n_noted"] = int((exp & fl.ne("normal").to_numpy()).sum())
    t = _col(df, "bjd")
    out["span_h"] = float((t.max() - t.min()) * 24.0) if t.notna().sum() > 1 else float("nan")
    am = _col(df, "airmass")
    out["airmass"] = (float(am.min()), float(am.max())) if am.notna().any() else (float("nan"), float("nan"))
    r = _col(df, "aperture_r_px")
    out["r_ap_med"] = float(r.median()) if r.notna().any() else float("nan")
    # night FWHM and k of this target: aperture_per_target.json (OUTPUT-REVIEW step 6); the LC
    # of older drafts repeats them on every row
    fwhm_night, k_star = _aperture_choice(Path(lc_csv))
    if not math.isfinite(fwhm_night):
        fw = _col(df, "fwhm_night_median_px")
        fwhm_night = float(fw.median()) if fw.notna().any() else float("nan")
    if not math.isfinite(k_star):
        k = _col(df, "aperture_f")
        k_star = float(k.median()) if k.notna().any() else float("nan")
    out["fwhm_night"] = fwhm_night
    out["k"] = k_star
    jd = _col(df, "jd")
    out["date_ut"] = ""
    if jd.notna().any():
        from astropy.time import Time  # noqa: PLC0415

        out["date_ut"] = Time(float(jd.min()), format="jd", scale="utc").iso[:10]
    return out


# ---------------------------------------------------------------- ePSF page (EPSF-PDF-2026-10)
def psf_lc_header(psf_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with Path(psf_csv).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            k, sep, v = line[1:].strip().partition("=")
            if sep:
                out[k.strip()] = v.strip()
    return out


def epsf_page_data(lc_dir: Path, platesolve_dir: Path, cid: str, *, min_epochs: int = 5) -> dict[str, Any] | None:
    """Facts and series of the ePSF page of one target; None without a usable ePSF light curve.

    Series from ``epsf_lc_compare.load_pair`` (aperture and ePSF differential curves on the aperture
    epochs); statistics on the epochs both methods kept (``pair_stats``); epoch accounting from the
    ePSF light-curve header; unresolved companions and model summary from the RUN ePSF manifest.
    """
    from epsf_lc_compare import frame_fwhm_map, load_pair, pair_stats  # noqa: PLC0415
    from report_methods import lc_csv_path  # noqa: PLC0415

    psf_csv = lc_csv_path(lc_dir, cid, "psf")
    if not psf_csv.is_file() or not lc_csv_path(lc_dir, cid, "aperture").is_file():
        return None
    df = load_pair(lc_dir, cid, frame_fwhm_map(platesolve_dir))
    if int(df["psf_ok"].sum()) < int(min_epochs):
        return None
    st = pair_stats(df)
    hdr = psf_lc_header(psf_csv)
    man_p = Path(platesolve_dir) / "epsf_frames" / "epsf_frames_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.is_file() else {}
    n_ep = int(st["n_epochs"])

    def _i(key: str) -> int:
        try:
            return int(float(hdr.get(key, "0") or 0))
        except ValueError:
            return 0

    drops = []
    for key, label in (("psf_lc_n_epochs_target_saturated", "the target saturated"),
                       ("psf_lc_n_epochs_target_outside_footprint", "the target off the frame's data"),
                       ("psf_lc_n_epochs_target_fit_not_ok", "the target PSF fit failed"),
                       ("psf_lc_n_epochs_dropped_pin", "the frame not fitted or a comparison star missing")):
        if _i(key):
            drops.append(f"{_i(key)} with {label}")
    facts: list[tuple[str, str]] = [
        ("ePSF epochs", f"{int(st['n_psf'])} of {n_ep}" + (f" epochs; dropped: {'; '.join(drops)}" if drops else "")),
        ("Common epochs", f"{int(st['n_both'])} (aperture and ePSF both measured)"),
    ]
    ap, ps = st["ap_p2p_mmag"], st["psf_p2p_mmag"]
    if math.isfinite(ap) and math.isfinite(ps) and ap > 0:
        facts.append(("Scatter p2p", f"aperture {ap:.1f} / ePSF {ps:.1f} mmag (ePSF/aperture {ps / ap:.2f}; "
                                     "common epochs)"))
    facts.append(("Median error", f"aperture {st['ap_err_med_mmag']:.1f} / ePSF {st['psf_err_med_mmag']:.1f} mmag"))
    if math.isfinite(st["diff_p2p_mmag"]):
        line = f"p2p {st['diff_p2p_mmag']:.1f} mmag"
        if math.isfinite(st["diff_fwhm_slope_mmag_px"]):
            line += f", vs seeing {st['diff_fwhm_slope_mmag_px']:+.1f} mmag/px"
        facts.append(("ePSF - aperture", line))
    if hdr.get("ensemble_n_comp"):
        # REPORT-AUDIT-2026-10: the ePSF ensemble is the aperture ensemble minus the comparison stars
        # without an ePSF measurement on this night
        _drop = [x for x in str(hdr.get("ensemble_dropped_no_psf_measurement") or "").split(",") if x.strip()]
        facts.append(("Comparison stars", f"{hdr['ensemble_n_comp']} of the target's aperture ensemble"
                      + (f" ({len(_drop)} without ePSF measurement dropped)" if _drop else " (all)")))
    merged = [m for m in ((man.get("fit_sources") or {}).get("merged_neighbours") or []) if str(m.get("catalog_id")) == str(cid)]
    if merged:
        txt = "; ".join(f"Gaia DR3 {m['neighbour_id']} at {float(m['sep_px']):.1f} px, {float(m['dmag']):+.2f} mag "
                        f"({100 * 10 ** (-0.4 * float(m['dmag'])):.1f} % of the light, kept with the target)" for m in merged)
        facts.append(("Unresolved neighbour", txt))
    frames = man.get("frames") or []
    if frames:
        fw = np.array([float(f.get("fwhm_px", np.nan)) for f in frames])
        grid = "x".join(str(v) for v in ((man.get("spatial") or {}).get("grid") or [1, 1]))
        facts.append(("ePSF models", f"{len(frames)} frames, FWHM {np.nanmedian(fw):.2f} px ({np.nanmin(fw):.2f}-"
                                     f"{np.nanmax(fw):.2f}), grid {grid}, {man.get('n_build_stars', '?')} build stars"))
    return {"df": df, "stats": st, "facts": facts, "psf_csv": psf_csv}


def plot_lc_epsf(df: pd.DataFrame, out_png: Path, *, figsize: tuple[float, float] = (10.0, 5.6)) -> Path | None:
    """Aperture (light blue) and ePSF (orange diamonds) differential light curves on one axis - the ePSF
    curve at the aperture level (median over the common epochs; each ensemble has its own Broeg weights)
    - and the ePSF - aperture difference per epoch in a strip under it (mmag, same time axis)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = df.loc[df["ap_ok"]]
    ps = df.loc[df["psf_ok"]]
    both = df.loc[df["ap_ok"] & df["psf_ok"]]
    if len(ps) < 2:
        return None
    from epsf_lc_compare import psf_level_shift  # noqa: PLC0415

    shift = psf_level_shift(df)
    from jd_axis_format import jd_offset  # noqa: PLC0415

    x0 = float(jd_offset(df["bjd"]) or 0)
    fig = plt.figure(figsize=figsize, dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axd = fig.add_subplot(gs[1], sharex=ax)
    try:
        ax.errorbar(ap["bjd"] - x0, ap["ap_mag"], yerr=ap["ap_err"], fmt="o", ms=3.0,
                    color="#8fb8e6", ecolor="#d3e3f5", elinewidth=0.7, capsize=0, label="aperture", zorder=2)
        ax.errorbar(ps["bjd"] - x0, ps["psf_mag"] - shift, yerr=ps["psf_err"], fmt="D", ms=3.6, color="#e0601f",
                    ecolor="#f3b693", elinewidth=0.8, capsize=0, label="ePSF", zorder=3)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("differential magnitude", fontsize=9)
        ax.legend(fontsize=7.5, loc="best", frameon=False)
        d = 1e3 * (both["psf_mag"] - both["ap_mag"] - shift)
        axd.plot(both["bjd"] - x0, d, "o", ms=2.4, color="#6e7781")
        axd.axhline(0, color="#d0d7de", lw=0.8)
        lim = max(10.0, float(np.nanpercentile(np.abs(d), 98)) * 1.3) if len(d) else 10.0
        axd.set_ylim(-lim, lim)
        axd.grid(True, alpha=0.3, lw=0.5)
        axd.tick_params(labelsize=7)
        axd.set_ylabel("ePSF - AP [mmag]", fontsize=8)
        plt.setp(ax.get_xticklabels(), visible=False)
        axd.set_xlabel(f"BJD_TDB - {x0:.0f}", fontsize=9)
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out_png
