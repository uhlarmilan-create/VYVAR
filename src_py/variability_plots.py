# -*- coding: ascii -*-
"""Figures of the field variability search, shared by the UI tab and the PDF report
(VAR-2026-10): hockey-stick plot, inverted field image with candidates, candidate LC."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

C_FIELD = "#7A8592"
C_REJ = "#D4D4D4"
C_TARGET = "#1F5FBF"
C_CAND = "#D9480F"
C_SYS = "#B08900"


def _mpl():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def noise_curves(field: pd.DataFrame, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Noise model and candidate limit of the scatter index along G (0.5 mag bins).

    Shared by the PDF figure and the interactive Variability tab.
    """
    f = field
    nm = meta.get("noise_model") or {}
    a, fl = float(nm.get("a", float("nan"))), float(nm.get("floor_mag", float("nan")))
    mag = pd.to_numeric(f["mag"], errors="coerce")
    e = pd.to_numeric(f["e"], errors="coerce")
    s = pd.to_numeric(f["s"], errors="coerce")
    g = f["gates_ok"].astype(bool)
    if not (math.isfinite(a) and math.isfinite(fl) and g.any()):
        return None
    ok = g & np.isfinite(mag) & np.isfinite(e)
    if not ok.any():
        return None
    bins = np.arange(np.floor(mag[ok].min()), np.ceil(mag[ok].max()) + 0.5, 0.5)
    mids, emed = [], []
    for b0 in bins:
        x = e[ok & (mag >= b0) & (mag < b0 + 0.5)]
        if len(x) >= 5:
            mids.append(b0 + 0.25)
            emed.append(float(np.median(x)))
    if not mids:
        return None
    mids_a = np.array(mids)
    sexp = np.sqrt((a * np.array(emed)) ** 2 + fl**2)
    zt = float(meta.get("z_threshold", float("nan")))
    ref = g & ~f["known_variable"].astype(bool)
    lq = np.log(s[ref] / np.sqrt((a * e[ref]) ** 2 + fl**2))
    lq = lq[np.isfinite(lq)]
    if not len(lq) or not math.isfinite(zt):
        return mids_a, sexp, None
    from variability_field import scatter_index_center_scale  # noqa: PLC0415

    md, sc = scatter_index_center_scale(lq.to_numpy(float))
    return mids_a, sexp, sexp * np.exp(md + zt * sc)


def hockey_png(field: pd.DataFrame, meta: dict[str, Any], out: Path, title: str = "") -> Path:
    """Robust scatter vs Gaia G of every star, noise model and candidate limit."""
    plt = _mpl()
    f = field.copy()
    for c in ("mag", "s", "e", "z_s"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    nm = meta.get("noise_model") or {}
    a, fl = float(nm.get("a", float("nan"))), float(nm.get("floor_mag", float("nan")))
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=130)
    g = f["gates_ok"].astype(bool)
    tgt = f["target"].astype(bool)
    kn = f["known_variable"].astype(bool) & ~tgt
    sysf = f["systematic"].astype(bool) & g
    cand = f["candidate"].astype(bool)
    ax.scatter(f.loc[~g, "mag"], f.loc[~g, "s"], s=5, c=C_REJ, lw=0, label="rejected by gates (blend / edge / SNR)")
    sel = g & ~tgt & ~kn & ~cand & ~sysf
    ax.scatter(f.loc[sel, "mag"], f.loc[sel, "s"], s=6, c=C_FIELD, lw=0, label="field stars")
    ax.scatter(f.loc[sysf, "mag"], f.loc[sysf, "s"], s=22, marker="^", facecolors="none", edgecolors=C_SYS, lw=1.0,
               label="signal explained by seeing / airmass")
    ax.scatter(f.loc[tgt, "mag"], f.loc[tgt, "s"], s=36, marker="x", c=C_TARGET, lw=1.2, label="targets")
    ax.scatter(f.loc[kn, "mag"], f.loc[kn, "s"], s=36, facecolors="none", edgecolors=C_TARGET, lw=1.2, label="known variable")
    ax.scatter(f.loc[cand, "mag"], f.loc[cand, "s"], s=60, c=C_CAND, edgecolors="k", lw=0.6, zorder=5, label="candidate")
    cur = noise_curves(f, meta)
    if cur is not None:
        mids, sexp, slim = cur
        ax.plot(mids, sexp, c="#222222", lw=1.6, label=f"noise model (a = {a:.2f}, floor = {1000 * fl:.1f} mmag)")
        if slim is not None:
            ax.plot(mids, slim, c=C_CAND, lw=1.4, ls="--",
                    label=f"candidate limit, scatter index (field FAP 1 %, z > {float(meta.get('z_threshold')):.2f})")
    for _, r in f[cand].iterrows():
        ax.annotate(str(r.get("label", "")), (r["mag"], r["s"]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=C_CAND)
    ax.set_yscale("log")
    ax.set_xlabel("Gaia G [mag]")
    ax.set_ylabel("robust scatter 1.4826 MAD [mag]")
    ax.grid(alpha=0.25, which="both")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out


def field_png(masterstar: Path, cand: pd.DataFrame, out: Path) -> Path | None:
    """Inverted MASTERSTAR (dark stars on white sky) with candidates only, N/E arrows."""
    from astropy.io import fits  # noqa: PLC0415
    from astropy.visualization import ZScaleInterval  # noqa: PLC0415
    from astropy.wcs import WCS  # noqa: PLC0415

    plt = _mpl()
    p = Path(masterstar)
    if not p.is_file():
        return None
    with fits.open(p, memmap=False) as h:
        img = np.asarray(h[0].data, dtype=float)
        hdr = h[0].header
    lo, hi = ZScaleInterval().get_limits(img[np.isfinite(img)])
    ny, nx = img.shape
    fig = plt.figure(figsize=(8, 8 * ny / nx), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1), cmap="gray_r", origin="lower")
    rad = 0.012 * nx
    for _, r in cand.iterrows():
        x, y = float(r["x"]), float(r["y"])
        ax.add_patch(plt.Circle((x, y), rad, fill=False, ec=C_CAND, lw=2))
        ax.text(x + 1.2 * rad, y + 1.2 * rad, str(r.get("label", "")), color=C_CAND, fontsize=12, weight="bold")
    try:
        w = WCS(hdr)
        if w.has_celestial:
            x0, y0 = 0.08 * nx, 0.10 * ny
            ra0, de0 = w.all_pix2world(x0, y0, 0)
            L = 0.06 * nx
            for lab, (dra, dde) in (("N", (0.0, 1.0)), ("E", (1.0, 0.0))):
                scale = abs(w.proj_plane_pixel_scales()[0].value)
                x1, y1 = w.all_world2pix(ra0 + dra * L * scale / max(math.cos(math.radians(float(de0))), 1e-6), de0 + dde * L * scale, 0)
                ax.annotate("", xy=(float(x1), float(y1)), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#1B1F24", lw=1.5))
                ax.text(float(x1), float(y1), lab, fontsize=11, color="#1B1F24")
    except Exception:  # noqa: BLE001
        pass
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_axis_off()
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out


def candidate_lc_png(lc_csv: Path | pd.DataFrame, out: Path, title: str, time_label: str = "JD",
                     ylabel: str = "dmag") -> Path | None:
    """Night light curve of one star with FWHM and airmass under it. Error bars: the
    field noise model ((a e)^2 + floor^2), the scatter a constant star of this brightness has.
    ``lc_csv``: the light-curve CSV or the table itself (G11-44: no temporary file)."""
    plt = _mpl()
    if isinstance(lc_csv, pd.DataFrame):
        d = lc_csv
    else:
        p = Path(lc_csv)
        if not p.is_file():
            return None
        d = pd.read_csv(p)
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.6), dpi=110, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    t = pd.to_numeric(d["time"], errors="coerce")
    from jd_axis_format import jd_offset  # noqa: PLC0415

    t0 = jd_offset(t) or 0
    axs[0].errorbar(t - t0, d["dmag"], yerr=d["err"], fmt="o", ms=3, c="#1B1F24", ecolor="#9AA3AE", lw=0.8)
    axs[0].invert_yaxis()
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(title, fontsize=10)
    axs[1].plot(t - t0, d["fwhm"], c="#5A6370", lw=1, label="FWHM [px]")
    if "airmass" in d and pd.to_numeric(d["airmass"], errors="coerce").notna().any():
        ax2 = axs[1].twinx()
        ax2.plot(t - t0, d["airmass"], c="#B08900", lw=1)
        ax2.set_ylabel("airmass", fontsize=8)
    axs[1].set_ylabel("FWHM", fontsize=8)
    tl = {"bjd_tdb": "BJD_TDB", "bjd_tdb_mid": "BJD_TDB", "jd_mid": "JD (UTC)"}.get(str(time_label), str(time_label))
    axs[1].set_xlabel(f"{tl} - {t0:.0f} [d]")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out
