# -*- coding: ascii -*-
"""EPSF-SHAPE-01: why R-A2-3 (VYVAR ePSF vs PSFEx) disagrees.

Dev-only. src_py must not import this module. Linux a2/ is read-only
and gitignored. Production ePSF is loaded the same way photometry
does (fits.getdata + meta oversampling; psf_photometry.py:2817-2818).
FWHM uses a verbatim copy of _epsf_fwhm_native_from_profile
(psf_photometry.py:527-569).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats
from scipy.ndimage import shift as nd_shift

REPO = Path(__file__).resolve().parents[2]
SESSION_A2 = REPO / "dev" / "results" / "context" / "session_20260907_epsfxval_a2"
A2_OUT = SESSION_A2 / "a2" / "out"
A2_COMPARE = SESSION_A2 / "a2_compare"
VYREF = SESSION_A2 / "vyvar_reference"
OUT = REPO / "dev" / "results" / "context" / "session_20260914_epsf_shape_01"
LIVE_PS = REPO / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2"
EPSF_FITS = LIVE_PS / "masterstar_epsf.fits"
EPSF_META = LIVE_PS / "masterstar_epsf_meta.json"
MS_PATH = LIVE_PS / "masterstars_full_match.csv"
QC_PATH = VYREF / "qc_metrics.csv"

TARGET_CID = "1498613634033133184"
CHECK_CID = "1497613731286514432"
ENS_IDS = [
    "1497771992240531712",
    "1499200223486564608",
    "1497974027502858240",
    "1497368849430107904",
]
STARS = [TARGET_CID, CHECK_CID] + ENS_IDS
PROBE_STEMS = [
    "BO_CVn_Light_001",
    "BO_CVn_Light_037",
    "BO_CVn_Light_076",
    "BO_CVn_Light_109",
    "BO_CVn_Light_148",
]
G4_EXPECT = {"csv": "bfa24039", "fits": "13e77cf8", "epsf": "172f9540"}
G4_PATHS = {
    "csv": MS_PATH,
    "fits": LIVE_PS / "MASTERSTAR.fits",
    "epsf": EPSF_FITS,
}
NATIVE_N = 31
ROLES = {
    TARGET_CID: "target",
    CHECK_CID: "check",
    ENS_IDS[0]: "ensemble",
    ENS_IDS[1]: "ensemble",
    ENS_IDS[2]: "ensemble",
    ENS_IDS[3]: "ensemble",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def g4_live_516() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key, path in G4_PATHS.items():
        digest = _sha256_file(path)
        out[key] = {
            "prefix": digest[:8],
            "verdict": "PASS" if digest.startswith(G4_EXPECT[key]) else "FAIL",
        }
    return out


def epsf_fwhm_native_from_profile(epsf_data: np.ndarray, *, osamp: int) -> float:
    """Verbatim of psf_photometry.py:527-569 (_epsf_fwhm_native_from_profile)."""
    z = np.asarray(epsf_data, dtype=np.float64)
    cy, cx = np.array(z.shape) // 2
    yy, xx = np.indices(z.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    v = z.ravel()
    ok = np.isfinite(v) & np.isfinite(r)
    r = r[ok]
    v = v[ok]
    if r.size < 10:
        return float("nan")
    peak = float(np.max(v))
    if peak <= 0:
        return float("nan")
    v = v / peak
    h, w = z.shape
    rmax = min(h, w) * 0.45
    bin_w = 0.5
    edges = np.arange(0.0, rmax + bin_w * 0.5, bin_w)
    centers: list[float] = []
    means: list[float] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (r >= lo) & (r < hi)
        if int(sel.sum()) < 3:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(float(np.mean(v[sel])))
    if len(centers) < 4:
        return float("nan")
    c_arr = np.asarray(centers, dtype=np.float64)
    m_arr = np.asarray(means, dtype=np.float64)
    cross = None
    for i in range(len(c_arr) - 1):
        a, b = m_arr[i], m_arr[i + 1]
        if a >= 0.5 >= b and a != b:
            frac = (a - 0.5) / (a - b)
            cross = float(c_arr[i] + frac * (c_arr[i + 1] - c_arr[i]))
            break
    if cross is None:
        return float("nan")
    return float(2.0 * cross / max(1, int(osamp)))


def load_vyvar_epsf() -> tuple[np.ndarray, dict]:
    """Production read: psf_photometry.py:2767-2818 (fits.getdata + meta osamp)."""
    meta = json.loads(EPSF_META.read_text(encoding="utf-8"))
    arr = np.asarray(fits.getdata(EPSF_FITS), dtype=np.float64)
    osamp = int(meta.get("oversampling", 2))
    # Production save-time norm: arr.sum() / osamp^2 == epsf_sum_native (line 649-661).
    return arr, {
        "oversampling": osamp,
        "cutout_size": int(meta.get("cutout_size", 17)),
        "fwhm_px_meta": float(meta.get("fwhm_px", float("nan"))),
        "spatial_order": int(meta.get("spatial_order", -1)),
        "n_stars_used": int(meta.get("n_stars_used", -1)),
        "epsf_sum_native": float(meta.get("epsf_sum_native", float("nan"))),
        "epsf_fwhm_qc": float((meta.get("epsf_qc") or {}).get("epsf_fwhm_native_px") or float("nan")),
        "created_utc": str(meta.get("created_utc", "")),
    }


def downsample_osamp(arr: np.ndarray, osamp: int) -> np.ndarray:
    """Native-pixel view of an oversampled ePSF.

    Production unit-sum convention (psf_photometry.py:649-661):
    epsf_sum_native = sum(oversampled) / osamp^2. A native pixel is the
    sum of the osamp x osamp oversampled block.
    """
    z = np.asarray(arr, dtype=np.float64)
    osamp = max(1, int(osamp))
    if osamp == 1:
        return z
    h, w = z.shape
    h2 = (h // osamp) * osamp
    w2 = (w // osamp) * osamp
    y0 = (h - h2) // 2
    x0 = (w - w2) // 2
    crop = z[y0 : y0 + h2, x0 : x0 + w2]
    return crop.reshape(h2 // osamp, osamp, w2 // osamp, osamp).sum(axis=(1, 3))


def embed_center(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=np.float64)
    a = np.asarray(arr, dtype=np.float64)
    ah, aw = a.shape
    y0 = (n - ah) // 2
    x0 = (n - aw) // 2
    ys = max(0, y0)
    xs = max(0, x0)
    ye = min(n, y0 + ah)
    xe = min(n, x0 + aw)
    asy = ys - y0
    asx = xs - x0
    out[ys:ye, xs:xe] = a[asy : asy + (ye - ys), asx : asx + (xe - xs)]
    return out


def unit_sum(arr: np.ndarray) -> np.ndarray:
    z = np.asarray(arr, dtype=np.float64)
    s = float(np.nansum(z))
    if not (math.isfinite(s) and s > 0):
        return z
    return z / s


def bilinear_sample(arr: np.ndarray, y: float, x: float) -> float:
    h, w = arr.shape
    if y < 0 or x < 0 or y > h - 1 or x > w - 1:
        return 0.0
    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    y1 = min(h - 1, y0 + 1)
    x1 = min(w - 1, x0 + 1)
    wy = y - y0
    wx = x - x0
    v00 = float(arr[y0, x0])
    v01 = float(arr[y0, x1])
    v10 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])
    return (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)


def resample_to_native_grid(arr: np.ndarray, src_scale: float, n: int = NATIVE_N) -> np.ndarray:
    """arr is sampled at src_scale native-px per array pixel; centre -> centre."""
    a = np.asarray(arr, dtype=np.float64)
    h, w = a.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    oc = (n - 1) / 2.0
    iy, ix = np.mgrid[0:n, 0:n]
    ys = cy + (iy.astype(np.float64) - oc) / float(src_scale)
    xs = cx + (ix.astype(np.float64) - oc) / float(src_scale)
    out = np.zeros((n, n), dtype=np.float64)
    ok = (ys >= 0) & (xs >= 0) & (ys <= h - 1) & (xs <= w - 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(h - 1, y0 + 1)
    x1 = np.minimum(w - 1, x0 + 1)
    wy = ys - y0
    wx = xs - x0
    y0c = np.clip(y0, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    y1c = np.clip(y1, 0, h - 1)
    x1c = np.clip(x1, 0, w - 1)
    v00 = a[y0c, x0c]
    v01 = a[y0c, x1c]
    v10 = a[y1c, x0c]
    v11 = a[y1c, x1c]
    out[ok] = (
        (1 - wy[ok]) * ((1 - wx[ok]) * v00[ok] + wx[ok] * v01[ok])
        + wy[ok] * ((1 - wx[ok]) * v10[ok] + wx[ok] * v11[ok])
    )
    return out


class PsfexModel:
    """Minimal AstrOmatic PSFEx reader (Bertin 2011). FITS PSF_DATA bintable."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with fits.open(self.path) as hdul:
            hdu = hdul["PSF_DATA"]
            hdr = hdu.header
            self.polzero1 = float(hdr["POLZERO1"])
            self.polzero2 = float(hdr["POLZERO2"])
            self.polscal1 = float(hdr["POLSCAL1"])
            self.polscal2 = float(hdr["POLSCAL2"])
            self.poldeg = int(hdr["POLDEG1"])
            self.psf_fwhm = float(hdr["PSF_FWHM"])
            self.psf_samp = float(hdr["PSF_SAMP"])
            self.psfnaxis = int(hdr["PSFNAXIS"])
            mask = np.asarray(hdu.data["PSF_MASK"][0], dtype=np.float64)
        if mask.ndim != 3:
            raise ValueError(f"{self.path.name}: PSF_MASK ndim={mask.ndim}")
        # TDIM (31,31,6) -> numpy (6,31,31)
        if mask.shape[0] not in (self._ncoeff(),) and mask.shape[-1] == self._ncoeff():
            mask = np.moveaxis(mask, -1, 0)
        self.basis = mask
        if self.basis.shape[0] != self._ncoeff():
            raise ValueError(
                f"{self.path.name}: ncoeff={self._ncoeff()} vs basis {self.basis.shape}"
            )

    def _ncoeff(self) -> int:
        d = int(self.poldeg)
        return (d + 1) * (d + 2) // 2

    def _basis_weights(self, x: float, y: float) -> np.ndarray:
        u = (float(x) - self.polzero1) / self.polscal1
        v = (float(y) - self.polzero2) / self.polscal2
        w = []
        for deg in range(self.poldeg + 1):
            for p in range(deg + 1):
                w.append((u ** (deg - p)) * (v ** p))
        return np.asarray(w, dtype=np.float64)

    def reconstruct(self, x: float, y: float) -> np.ndarray:
        w = self._basis_weights(x, y)
        out = np.zeros(self.basis.shape[1:], dtype=np.float64)
        for i, wi in enumerate(w):
            out += float(wi) * self.basis[i]
        return out

    def native_stamp(self, x: float, y: float, n: int = NATIVE_N) -> np.ndarray:
        rec = self.reconstruct(x, y)
        return unit_sum(resample_to_native_grid(rec, src_scale=self.psf_samp, n=n))


def _pipeline_meta_spatial_flags() -> dict:
    """Live 516 pipeline_meta.json: spatial ePSF was OFF."""
    pm = json.loads((LIVE_PS / "photometry" / "pipeline_meta.json").read_text(encoding="utf-8"))
    cfg = {}
    if isinstance(pm, dict):
        # config blob is nested; walk one level of dicts
        stack = [pm]
        while stack:
            cur = stack.pop()
            if not isinstance(cur, dict):
                continue
            if "psf_spatial_enabled" in cur:
                cfg = cur
                break
            stack.extend(v for v in cur.values() if isinstance(v, dict))
    return {
        "psf_spatial_enabled": bool(cfg.get("psf_spatial_enabled", True)),
        "psf_spatial_order": int(cfg.get("psf_spatial_order", -1)),
        "psf_spatial_grid": str(cfg.get("psf_spatial_grid", "")),
    }


def verify_p1(meta: dict) -> dict:
    """Single global ePSF; grid path gated/offline and unused on 516."""
    plan = json.loads((LIVE_PS / "photometry_plan.json").read_text(encoding="utf-8"))
    # comparison-star grid is unrelated (photometry_plan comparison_selection).
    grid_files = list(LIVE_PS.glob("*epsf*grid*")) + list(LIVE_PS.glob("*grid*epsf*"))
    flags = _pipeline_meta_spatial_flags()
    p1 = {
        "spatial_order": meta["spatial_order"],
        "n_stars_used": meta["n_stars_used"],
        "oversampling": meta["oversampling"],
        "sha_prefix": _sha256_file(EPSF_FITS)[:8],
        "n_grid_files": len(grid_files),
        "plan_grid_nx": int((plan.get("comparison_selection") or {}).get("grid_nx", 0) or 0),
        **flags,
    }
    if meta["spatial_order"] != 0:
        raise SystemExit(f"STOP P1: spatial_order={meta['spatial_order']} (expected 0)")
    if meta["n_stars_used"] != 67:
        raise SystemExit(f"STOP P1: n_stars_used={meta['n_stars_used']} (expected 67)")
    if p1["sha_prefix"] != "172f9540":
        raise SystemExit(f"STOP P1: ePSF sha {p1['sha_prefix']} != 172f9540")
    if flags["psf_spatial_enabled"]:
        raise SystemExit("STOP P1: pipeline_meta psf_spatial_enabled is true")
    if grid_files:
        raise SystemExit(f"STOP P1: grid products present {grid_files}")
    return p1


def _parse_psfex_diag_fwhm(stem_dir: Path) -> float:
    """PSFEx writes the diagnostic table to stderr (stdout is empty)."""
    err_path = stem_dir / "psfex.stderr"
    if not err_path.is_file():
        return float("nan")
    text = err_path.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(
        r"(\d+)/(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        text,
    )
    if not hits:
        return float("nan")
    return float(hits[-1][4])


def validate_reader(work: Path) -> tuple[list[dict], str]:
    """Reader integrity before any metric.

    Header PSF_FWHM is 4.7 * PSF_SAMP on every probe frame (sampling
    design FWHM, Bertin PSFEx automatic sampling). That is not a
    measured width of PSF_MASK. psfex.stdout is empty; the console
    diagnostic FWHM is on psfex.stderr and uses a different estimator
    than _epsf_fwhm_native_from_profile. The 5% reconstructed-vs-header
    gate in the task brief is therefore a category error (architect
    error 21). Integrity gates that do fire: center==component-0,
    header/samp == 4.7 within 5%, degree-2 corners are not constant.
    """
    rows = []
    for stem in PROBE_STEMS:
        psf_path = work / stem / f"{stem}.psf"
        if not psf_path.is_file():
            raise SystemExit(f"STOP reader: missing {psf_path}")
        model = PsfexModel(psf_path)
        rec = model.reconstruct(model.polzero1, model.polzero2)
        d0 = float(np.nanmax(np.abs(rec - model.basis[0])))
        if d0 > 1e-12:
            raise SystemExit(f"STOP reader: {stem} center != component 0 (maxabs={d0})")
        design = model.psf_fwhm / model.psf_samp if model.psf_samp else float("nan")
        if not (math.isfinite(design) and abs(design - 4.7) / 4.7 <= 0.05):
            raise SystemExit(
                f"STOP reader: {stem} PSF_FWHM/PSF_SAMP={design} (expected 4.7)"
            )
        fwhm = epsf_fwhm_native_from_profile(rec, osamp=1) * model.psf_samp
        hdr = model.psf_fwhm
        rel_hdr = abs(fwhm - hdr) / hdr if hdr > 0 else float("nan")
        diag = _parse_psfex_diag_fwhm(work / stem)
        rel_diag = abs(fwhm - diag) / diag if (math.isfinite(diag) and diag > 0) else float("nan")
        c1 = model.reconstruct(
            model.polzero1 - 0.45 * model.polscal1,
            model.polzero2 - 0.45 * model.polscal2,
        )
        c2 = model.reconstruct(
            model.polzero1 + 0.45 * model.polscal1,
            model.polzero2 + 0.45 * model.polscal2,
        )
        vary = float(np.nanmax(np.abs(c1 - c2)))
        const0 = float(np.nanmax(np.abs(c1 - rec)))
        stdout_p = work / stem / "psfex.stdout"
        row = {
            "stem": stem,
            "psf_fwhm_header": hdr,
            "psf_samp": model.psf_samp,
            "header_over_samp": design,
            "fwhm_reconstructed_radial": fwhm,
            "rel_err_vs_header": rel_hdr,
            "psfex_stderr_diag_fwhm": diag,
            "rel_err_vs_stderr_diag": rel_diag,
            "corner_delta_max": vary,
            "center_vs_corner_max": const0,
            "center_eq_c0_maxabs": d0,
            "poldeg": model.poldeg,
            "ncoeff": model.basis.shape[0],
            "stdout_empty": (not stdout_p.is_file()) or stdout_p.stat().st_size == 0,
        }
        if vary <= 1e-12:
            raise SystemExit(f"STOP reader: {stem} degree-2 reconstruction is constant")
        rows.append(row)
    note = (
        "architect error 21: task 5% gate vs header PSF_FWHM is a category "
        "error (header = 4.7*PSF_SAMP design FWHM; radial half-max of "
        "component 0 is ~24% below header and ~8-12% above stderr diagnostic). "
        "Reader algebra PASS (center==c0, corners vary, 4.7 identity)."
    )
    return rows, note


def ellipticity(arr: np.ndarray) -> float:
    z = np.asarray(arr, dtype=np.float64)
    z = np.where(np.isfinite(z) & (z > 0), z, 0.0)
    s = float(z.sum())
    if s <= 0:
        return float("nan")
    yy, xx = np.indices(z.shape)
    cy = float((z * yy).sum() / s)
    cx = float((z * xx).sum() / s)
    x = xx - cx
    y = yy - cy
    mxx = float((z * x * x).sum() / s)
    myy = float((z * y * y).sum() / s)
    mxy = float((z * x * y).sum() / s)
    tmp = math.sqrt((mxx - myy) ** 2 + 4.0 * mxy * mxy)
    a2 = 0.5 * ((mxx + myy) + tmp)
    b2 = 0.5 * ((mxx + myy) - tmp)
    if a2 <= 0 or b2 < 0:
        return float("nan")
    a = math.sqrt(a2)
    b = math.sqrt(max(b2, 0.0))
    if a <= 0:
        return float("nan")
    return float(1.0 - b / a)


def ee_within(arr: np.ndarray, radius: float) -> float:
    z = np.asarray(arr, dtype=np.float64)
    cy, cx = np.array(z.shape) // 2
    yy, xx = np.indices(z.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sel = r <= float(radius)
    tot = float(np.nansum(z))
    if tot <= 0:
        return float("nan")
    return float(np.nansum(z[sel]) / tot)


def _stamp_centroid(z: np.ndarray) -> tuple[float, float]:
    zz = np.where(np.isfinite(z) & (z > 0), z, 0.0)
    s = float(zz.sum())
    if s <= 0:
        h, w = z.shape
        return (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices(zz.shape)
    return float((zz * yy).sum() / s), float((zz * xx).sum() / s)


def resmap_rms(a: np.ndarray, b: np.ndarray, radius: float) -> float:
    """Normalized residual RMS inside radius after flux + centroid match."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    sa = float(np.nansum(aa))
    sb = float(np.nansum(bb))
    if sa <= 0 or sb <= 0:
        return float("nan")
    aa = aa / sa
    bb = bb / sb
    cya, cxa = _stamp_centroid(aa)
    cyb, cxb = _stamp_centroid(bb)
    bb = nd_shift(bb, shift=(cya - cyb, cxa - cxb), order=1, mode="constant", cval=0.0)
    sb2 = float(np.nansum(bb))
    if sb2 > 0:
        bb = bb / sb2
    cy, cx = np.array(aa.shape) // 2
    yy, xx = np.indices(aa.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sel = r <= float(radius)
    d = (aa - bb)[sel]
    d = d[np.isfinite(d)]
    if d.size < 5:
        return float("nan")
    return float(np.sqrt(np.mean(d * d)))


def load_qc() -> dict[str, float]:
    qc = pd.read_csv(QC_PATH, comment="#")
    out: dict[str, float] = {}
    for _, r in qc.iterrows():
        src = str(r.get("src") or "")
        m = re.search(r"(BO_CVn_Light_\d+)", src.replace("\\", "/"))
        if not m:
            continue
        out[m.group(1)] = float(pd.to_numeric(r.get("fwhm_px"), errors="coerce"))
    return out


def load_positions(label: str) -> pd.DataFrame:
    """a2_compare matched pass2 XPSF_IMAGE/YPSF_IMAGE (local match_rows; not committed)."""
    path = A2_COMPARE / f"match_rows_{label}.csv"
    if not path.is_file():
        raise SystemExit(f"STOP: {path} missing (needed for matched XPSF/YPSF)")
    df = pd.read_csv(path, dtype={"catalog_id": str})
    df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
    if df["matched"].dtype == object:
        df["matched"] = df["matched"].astype(str).str.lower().isin(("true", "1", "yes"))
    return df[df["catalog_id"].isin(STARS)].copy()


def vyvar_native_stamp(vy: np.ndarray, osamp: int) -> np.ndarray:
    native = downsample_osamp(vy, osamp)
    return unit_sum(embed_center(native, NATIVE_N))


def shape_rows(
    *,
    label: str,
    work: Path,
    pos: pd.DataFrame,
    vy_os: np.ndarray,
    osamp: int,
    fitrad: float,
    qc: dict[str, float],
) -> pd.DataFrame:
    vy_n = vyvar_native_stamp(vy_os, osamp)
    fwhm_vy = epsf_fwhm_native_from_profile(vy_os, osamp=osamp)
    e_vy = ellipticity(vy_n)
    ee_vy = ee_within(vy_n, fitrad)
    rows = []
    stems = sorted(pos["stem"].unique())
    for stem in stems:
        psf_path = work / stem / f"{stem}.psf"
        if not psf_path.is_file():
            continue
        model = PsfexModel(psf_path)
        sub = pos[(pos["stem"] == stem) & (pos["matched"])]
        for cid in STARS:
            hit = sub[sub["catalog_id"] == cid]
            if hit.empty:
                continue
            x = float(hit.iloc[0]["XPSF_IMAGE"])
            y = float(hit.iloc[0]["YPSF_IMAGE"])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            px_n = model.native_stamp(x, y)
            rec = model.reconstruct(x, y)
            fwhm_px = epsf_fwhm_native_from_profile(rec, osamp=1) * model.psf_samp
            ee_px = ee_within(px_n, fitrad)
            ratio = ee_vy / ee_px if (math.isfinite(ee_vy) and math.isfinite(ee_px) and ee_px > 0) else float("nan")
            mmag = -2.5 * math.log10(ratio) * 1000.0 if (math.isfinite(ratio) and ratio > 0) else float("nan")
            rows.append(
                {
                    "stem": stem,
                    "catalog_id": cid,
                    "role": ROLES[cid],
                    "x_psfex": x,
                    "y_psfex": y,
                    "qc_fwhm_px": qc.get(stem, float("nan")),
                    "fwhm_psfex": fwhm_px,
                    "fwhm_vyvar": fwhm_vy,
                    "e_psfex": ellipticity(px_n),
                    "e_vyvar": e_vy,
                    "ee_vyvar": ee_vy,
                    "ee_psfex": ee_px,
                    "ee_ratio_mmag": mmag,
                    "resmap_rms": resmap_rms(vy_n, px_n, fitrad),
                    "psfex_psf_fwhm_header": model.psf_fwhm,
                    "psf_samp": model.psf_samp,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"STOP: no shape rows for {label}")
    df.to_csv(OUT / f"shape_metrics_{label}.csv", index=False)
    return df


def spearman_theil(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[ok], yy[ok]
    n = int(xx.size)
    if n < 8:
        return {"n": float(n), "rho": float("nan"), "p": float("nan"), "slope": float("nan"), "r2_rank": float("nan")}
    rho, p = stats.spearmanr(xx, yy)
    sl, *_ = stats.theilslopes(yy, xx)
    rho_f = float(rho)
    return {
        "n": float(n),
        "rho": rho_f,
        "p": float(p),
        "slope": float(sl),
        "r2_rank": float(rho_f * rho_f) if math.isfinite(rho_f) else float("nan"),
    }


def irls_plane(x: np.ndarray, y: np.ndarray, d: np.ndarray) -> tuple[float, float, float]:
    """Huber IRLS for d ~ a + b x + c y."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    dd = np.asarray(d, dtype=np.float64)
    ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(dd)
    xx, yy, dd = xx[ok], yy[ok], dd[ok]
    a = np.column_stack([np.ones(len(xx)), xx, yy])
    w = np.ones(len(xx))
    coef = np.zeros(3)
    for _ in range(20):
        aw = a * w[:, None]
        coef, *_ = np.linalg.lstsq(aw, dd * w, rcond=None)
        resid = dd - a.dot(coef)
        s = float(np.median(np.abs(resid))) * 1.4826
        if s <= 1e-12:
            break
        u = resid / (1.345 * s)
        w = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.abs(u))
    return float(coef[0]), float(coef[1]), float(coef[2])


def _mech_max(a: float, b: float) -> float:
    vals = [v for v in (a, b) if math.isfinite(v)]
    return max(vals) if vals else float("nan")


def reading(c1_t: float, c1_c: float, c2_t: float, c2_c: float) -> str:
    """Pre-registered readings. Mechanism R^2 = max over the star it concerns."""
    seeing = _mech_max(c1_t, c1_c)
    spatial = _mech_max(c2_t, c2_c)
    detail = (
        f"M-SEEING target={c1_t:.3f} check={c1_c:.3f}; "
        f"M-SPATIAL target={c2_t:.3f} check={c2_c:.3f}"
    )

    def _band(v: float) -> str:
        if not math.isfinite(v):
            return "na"
        if v >= 0.5:
            return "dom"
        if v >= 0.2:
            return "mid"
        return "low"

    sb, pb = _band(seeing), _band(spatial)
    if sb == "dom" and pb != "dom":
        star = "target" if c1_t >= c1_c else "check"
        return f"R-SH1: M-SEEING is DOMINANT (R^2={seeing:.3f} on {star}). {detail}"
    if pb == "dom" and sb != "dom":
        star = "target" if c2_t >= c2_c else "check"
        return f"R-SH1: M-SPATIAL is DOMINANT (R^2={spatial:.3f} on {star}). {detail}"
    if sb == "dom" and pb == "dom":
        return (
            f"R-SH2: MIXED; M-SEEING={seeing:.3f} and M-SPATIAL={spatial:.3f} "
            f"both >= 0.5; no dominance claim. {detail}"
        )
    if sb == "mid" and pb == "mid":
        return (
            f"R-SH2: MIXED; M-SEEING={seeing:.3f} and M-SPATIAL={spatial:.3f} "
            f"each in 0.2-0.5; no dominance claim. {detail}"
        )
    if sb == "low" and pb == "low":
        return (
            "R-SH3: nothing reaches 0.2 on either star -> shape unsupported as the "
            "LC-level driver; escalate suspect to EPSF-CORE-01 (fit machinery). "
            + detail
        )
    # one mid, one low: no numbered reading fires as written
    mid_name = "M-SEEING" if sb == "mid" else "M-SPATIAL"
    mid_v = seeing if sb == "mid" else spatial
    return (
        f"No numbered reading fires as written. {mid_name} reaches R^2={mid_v:.3f} "
        f"(0.2-0.5 band); the other mechanism is < 0.2. No dominance. {detail}"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    vy, meta = load_vyvar_epsf()
    p1 = verify_p1(meta)
    print("[shape] P1", p1)
    fwhm_self = epsf_fwhm_native_from_profile(vy, osamp=meta["oversampling"])
    print("[shape] VYVAR FWHM self", fwhm_self, "meta_qc", meta["epsf_fwhm_qc"])

    reader_rows, reader_note = validate_reader(A2_OUT / "work")
    print("[shape] reader", reader_rows)
    print("[shape] reader_note", reader_note)

    fitrad = float(meta["fwhm_px_meta"])
    qc = load_qc()
    pos2 = load_positions("deg2")
    pos3 = load_positions("deg3")
    print("[shape] position source: a2_compare match_rows_* XPSF_IMAGE/YPSF_IMAGE")

    m2t = pd.read_csv(A2_COMPARE / "m2_epochs_target.csv")
    m2c = pd.read_csv(A2_COMPARE / "m2_epochs_check.csv")
    m1 = pd.read_csv(A2_COMPARE / "m1_per_star.csv", dtype={"catalog_id": str})
    m1["catalog_id"] = m1["catalog_id"].astype(str).str.strip()

    sh2 = shape_rows(
        label="deg2",
        work=A2_OUT / "work",
        pos=pos2,
        vy_os=vy,
        osamp=meta["oversampling"],
        fitrad=fitrad,
        qc=qc,
    )
    sh3 = shape_rows(
        label="deg3",
        work=A2_OUT / "deg3" / "work",
        pos=pos3,
        vy_os=vy,
        osamp=meta["oversampling"],
        fitrad=fitrad,
        qc=qc,
    )

    def _sub(df: pd.DataFrame, cid: str) -> pd.DataFrame:
        return df[df["catalog_id"] == cid]

    tgt = _sub(sh2, TARGET_CID).merge(m2t[["stem", "resid_after_median"]], on="stem", how="inner")
    chk = _sub(sh2, CHECK_CID).merge(m2c[["stem", "resid_after_median"]], on="stem", how="inner")
    tgt["resid_mmag"] = tgt["resid_after_median"] * 1000.0
    chk["resid_mmag"] = chk["resid_after_median"] * 1000.0

    c1_t = spearman_theil(tgt["qc_fwhm_px"], tgt["resid_mmag"])
    c1_c = spearman_theil(chk["qc_fwhm_px"], chk["resid_mmag"])
    c2_t = spearman_theil(tgt["ee_ratio_mmag"], tgt["resid_mmag"])
    c2_c = spearman_theil(chk["ee_ratio_mmag"], chk["resid_mmag"])

    both = m2t.merge(m2c, on="stem", suffixes=("_t", "_c"))
    c4 = spearman_theil(both["resid_after_median_t"] * 1000.0, both["resid_after_median_c"] * 1000.0)

    m1_ok = m1[(m1["n_ok"] >= 100) & np.isfinite(m1["phot_g_mean_mag"])].copy()
    ms = pd.read_csv(MS_PATH, dtype={"catalog_id": str})
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    m1_ok = m1_ok.merge(ms[["catalog_id", "x", "y"]], on="catalog_id", how="left")
    a, b, c = irls_plane(m1_ok["x"], m1_ok["y"], m1_ok["median_d_mag"] * 1000.0)
    pred = a + b * m1_ok["x"].to_numpy() + c * m1_ok["y"].to_numpy()
    amp = float(np.nanmax(pred) - np.nanmin(pred)) if len(pred) else float("nan")

    def _pred_at(cid: str) -> float:
        hit = ms[ms["catalog_id"] == cid]
        if hit.empty:
            return float("nan")
        return float(a + b * float(hit.iloc[0]["x"]) + c * float(hit.iloc[0]["y"]))

    pred_t = _pred_at(TARGET_CID)
    pred_c = _pred_at(CHECK_CID)
    obs_t = float(m1.loc[m1["catalog_id"] == TARGET_CID, "median_d_mag"].iloc[0]) * 1000.0
    obs_c = float(m1.loc[m1["catalog_id"] == CHECK_CID, "median_d_mag"].iloc[0]) * 1000.0

    def _top8(df: pd.DataFrame) -> set[str]:
        sub = df.reindex(df["resid_after_median"].abs().sort_values(ascending=False).index)
        return set(sub["stem"].head(8).astype(str))

    top_t = _top8(m2t)
    top_c = _top8(m2c)
    top_both = sorted(top_t & top_c)

    fwhm_vy = float(np.nanmedian(sh2["fwhm_vyvar"]))
    fwhm_px_t = float(np.nanmedian(_sub(sh2, TARGET_CID)["fwhm_psfex"]))
    fwhm_px_c = float(np.nanmedian(_sub(sh2, CHECK_CID)["fwhm_psfex"]))
    t_f = _sub(sh2, TARGET_CID).set_index("stem")["fwhm_psfex"]
    c_f = _sub(sh2, CHECK_CID).set_index("stem")["fwhm_psfex"]
    common_stems = t_f.index.intersection(c_f.index)
    spatial = float(np.nanmedian((c_f.loc[common_stems] - t_f.loc[common_stems]).to_numpy()))
    seeing = sh2.drop_duplicates("stem")["qc_fwhm_px"]
    seeing_spread = float(np.nanstd(seeing.to_numpy()))
    pct_t = 100.0 * (fwhm_vy - fwhm_px_t) / fwhm_px_t if fwhm_px_t else float("nan")
    pct_c = 100.0 * (fwhm_vy - fwhm_px_c) / fwhm_px_c if fwhm_px_c else float("nan")

    n1t, n1c = int(c1_t["n"]), int(c1_c["n"])
    n2t, n2c = int(c2_t["n"]), int(c2_c["n"])
    n4 = int(c4["n"])
    corr_rows = [
        {
            "test": "C1",
            "star": "target",
            "side": "M-SEEING",
            "population": (
                f"target {TARGET_CID}; {n1t} identical-ensemble epochs with "
                "finite resid_after_median and qc fwhm_px; resid vs qc fwhm_px"
            ),
            **c1_t,
        },
        {
            "test": "C1",
            "star": "check",
            "side": "M-SEEING",
            "population": (
                f"check {CHECK_CID}; {n1c} identical-ensemble epochs with "
                "finite resid_after_median and qc fwhm_px; resid vs qc fwhm_px"
            ),
            **c1_c,
        },
        {
            "test": "C2",
            "star": "target",
            "side": "M-SPATIAL",
            "population": (
                f"target {TARGET_CID}; {n2t} epochs; resid vs MISMATCH(f,target) "
                "ee_ratio_mmag"
            ),
            **c2_t,
        },
        {
            "test": "C2",
            "star": "check",
            "side": "M-SPATIAL",
            "population": (
                f"check {CHECK_CID}; {n2c} epochs; resid vs MISMATCH(f,check) "
                "ee_ratio_mmag"
            ),
            **c2_c,
        },
        {
            "test": "C4",
            "star": "both",
            "side": "common-mode",
            "population": (
                f"target vs check resid_after_median; {n4} shared identical-ensemble "
                f"epochs; top-8 |resid| overlap={top_both}"
            ),
            **c4,
        },
    ]
    pd.DataFrame(corr_rows).to_csv(OUT / "correlations.csv", index=False)

    fired = reading(c1_t["r2_rank"], c1_c["r2_rank"], c2_t["r2_rank"], c2_c["r2_rank"])
    standalone = []
    if abs(pct_t) > 5 or abs(pct_c) > 5:
        standalone.append(
            f"headline FWHM difference exceeds 5% (target {pct_t:.2f}%, check {pct_c:.2f}%)"
        )
    if math.isfinite(spatial) and math.isfinite(seeing_spread) and abs(spatial) > seeing_spread:
        standalone.append(
            f"spatial FWHM spread {spatial:.4f} px exceeds frame-to-frame seeing std {seeing_spread:.4f} px"
        )

    headline = {
        "p1": p1,
        "fitrad_px": fitrad,
        "fitrad_source": "masterstar_epsf_meta.json fwhm_px (A1 fitrad)",
        "fwhm_vyvar": fwhm_vy,
        "fwhm_vyvar_self_check": fwhm_self,
        "fwhm_psfex_target_median": fwhm_px_t,
        "fwhm_psfex_check_median": fwhm_px_c,
        "fwhm_pct_target": pct_t,
        "fwhm_pct_check": pct_c,
        "spatial_fwhm_check_minus_target_median": spatial,
        "seeing_fwhm_std_px": seeing_spread,
        "c1_target": c1_t,
        "c1_check": c1_c,
        "c2_target": c2_t,
        "c2_check": c2_c,
        "c4_common_mode": c4,
        "c3": {
            "population": (
                f"m1_per_star.csv stars with n_ok>=100 and finite G; n={int(len(m1_ok))}; "
                "d ~ a+bx+cy on masterstars_full_match.csv x,y (pixel); M1 is the "
                "65-star diagnostic, not the M2 product metric"
            ),
            "n_stars": int(len(m1_ok)),
            "a_mmag": a,
            "b_mmag_per_px": b,
            "c_mmag_per_px": c,
            "amplitude_mmag": amp,
            "pred_check_minus_target_mmag": pred_c - pred_t,
            "obs_m1_check_minus_target_mmag": obs_c - obs_t,
            "obs_m1_target_median_d_mmag": obs_t,
            "obs_m1_check_median_d_mmag": obs_c,
            "sign_match": bool(np.sign(pred_c - pred_t) == np.sign(obs_c - obs_t)),
            "position_source": "masterstars_full_match.csv x,y (same plane as the fit)",
        },
        "c4_top8_target": sorted(top_t),
        "c4_top8_check": sorted(top_c),
        "c4_top8_overlap": top_both,
        "reading": fired,
        "standalone": standalone,
        "reader_validation": reader_rows,
        "reader_note": reader_note,
        "g4": g4_live_516(),
        "position_source": "a2_compare match_rows_deg2/deg3 XPSF_IMAGE/YPSF_IMAGE",
        "n_shape_deg2": int(len(sh2)),
        "n_shape_deg3": int(len(sh3)),
    }
    (OUT / "headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    print("[shape] headline FWHM vy", fwhm_vy, "px_t", fwhm_px_t, "px_c", fwhm_px_c)
    print("[shape] spatial", spatial, "seeing_std", seeing_spread)
    print("[shape] C1 t/c", c1_t, c1_c)
    print("[shape] C2 t/c", c2_t, c2_c)
    print("[shape] C4", c4)
    print("[shape] C3", headline["c3"])
    print("[shape] reading", fired)
    print("[shape] standalone", standalone)
    print("[shape] g4", headline["g4"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
