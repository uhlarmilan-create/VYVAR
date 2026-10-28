# -*- coding: ascii -*-
"""EPSF-CHI2-LOCUS-01: data-derived chi2(flux) reference for psf_fit_ok (CHI2-FLUX-REF-2026-10).

Sets ``psf_fit_ok`` of the frame-level PSF fit (RUN ePSF); no fixed chi2 threshold.

chi2 of a PSF fit is a noise term (about constant for faint stars) plus a PSF-model-mismatch term that grows
roughly with flux^2, so log10(chi2) against log10(flux) is curved, not a straight line (DAOPHOT handles the same
effect with a profile-error term, Stetson 1987). A single straight line, set by the many faint stars, put the
bright, well-measured stars 4-8 sigma above it and rejected them (Brno 534 g: 35-56 % of stars brighter than
log10 flux 4.5 rejected while their PSF/aperture ratio was as stable as that of the accepted ones).

Night reference: the night's stars are pooled and binned in log10(flux) (``LOCUS_BIN_DEX``; consecutive bins are
merged until a group holds ``MIN_N_BIN`` stars); per group the plain median of log10(chi2) at the group's median log10(flux), and
the MAD sigma of the residuals from that curve; between group centres the curve is linear, outside it is extended linearly from the two
end groups (scatter held). A star is compared only with stars of similar flux.
Per frame, a plain median offset of log10(chi2) from the night curve (seeing changes chi2 for every star).
No clipping and no fitted functional form. Frames with fewer than ``MIN_N_LOCUS_FIT`` stars use the night curve
(source=night).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from gaia_catalog_id import read_vyvar_csv

LOGGER = logging.getLogger(__name__)

# Statistical sample sizes (not equipment): a frame offset needs MIN_N_LOCUS_FIT stars, a bin statistic MIN_N_BIN.
MIN_N_LOCUS_FIT = 10
MIN_N_BIN = 30
# Flux resolution of the reference: 0.25 dex = 0.625 mag.
LOCUS_BIN_DEX = 0.25
MAD_TO_SIGMA = 1.4826
DEFAULT_NSIGMA = 5.0


@dataclass(frozen=True)
class Chi2Locus:
    """Night chi2 reference curve (log10 flux bin centres, median log10 chi2, sigma) plus a frame offset."""

    centres: tuple[float, ...]
    med: tuple[float, ...]
    scatter: tuple[float, ...]
    n: int
    source: str  # "night" | "frame"
    offset: float = 0.0
    n_bin: tuple[int, ...] = field(default=())

    def predict(self, logf: np.ndarray) -> np.ndarray:
        """Linear between group centres, linear extrapolation from the two end groups outside."""
        x = np.asarray(logf, dtype=np.float64)
        c = np.asarray(self.centres, dtype=np.float64)
        m = np.asarray(self.med, dtype=np.float64)
        out = np.interp(x, c, m)
        if c.size >= 2:
            lo_s = (m[1] - m[0]) / (c[1] - c[0])
            hi_s = (m[-1] - m[-2]) / (c[-1] - c[-2])
            out = np.where(x < c[0], m[0] + lo_s * (x - c[0]), out)
            out = np.where(x > c[-1], m[-1] + hi_s * (x - c[-1]), out)
        return out + float(self.offset)

    def sigma(self, logf: np.ndarray) -> np.ndarray:
        return np.interp(logf, self.centres, self.scatter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "centres": [float(v) for v in self.centres],
            "med": [float(v) for v in self.med],
            "scatter": [float(v) for v in self.scatter],
            "n_bin": [int(v) for v in self.n_bin],
            "n": int(self.n),
            "source": self.source,
            "offset": float(self.offset),
        }


def _finite_mask(
    flux: np.ndarray,
    chi2: np.ndarray,
    converged: np.ndarray | None,
    saturated: np.ndarray | None,
) -> np.ndarray:
    f = np.asarray(flux, dtype=np.float64)
    c = np.asarray(chi2, dtype=np.float64)
    m = np.isfinite(f) & (f > 0) & np.isfinite(c) & (c > 0)
    if converged is not None:
        m = m & np.asarray(converged, dtype=bool)
    if saturated is not None:
        sat = np.asarray(saturated, dtype=bool)
        m = m & ~sat
    return m


def fit_chi2_locus(
    flux: Sequence[float] | np.ndarray,
    chi2: Sequence[float] | np.ndarray,
    *,
    converged: Sequence[bool] | np.ndarray | None = None,
    saturated: Sequence[bool] | np.ndarray | None = None,
    source: str = "night",
) -> Chi2Locus | None:
    """Binned reference curve: plain median and MAD sigma of log10(chi2) per log10(flux) bin.

    Returns None if the night has fewer than ``MIN_N_BIN`` stars.
    """
    f = np.asarray(flux, dtype=np.float64)
    c = np.asarray(chi2, dtype=np.float64)
    m = _finite_mask(f, c, converged, saturated)
    if int(m.sum()) < MIN_N_BIN:
        return None
    x = np.log10(f[m])
    y = np.log10(c[m])
    lo = math.floor(float(x.min()) / LOCUS_BIN_DEX) * LOCUS_BIN_DEX
    idx = np.floor((x - lo) / LOCUS_BIN_DEX).astype(int)
    # consecutive LOCUS_BIN_DEX bins are merged until a group holds MIN_N_BIN stars (sparse bright end);
    # a sparse remainder joins the previous group
    groups: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    for b in np.unique(idx):
        cur.append(np.flatnonzero(idx == b))
        if sum(len(v) for v in cur) >= MIN_N_BIN:
            groups.append(np.concatenate(cur))
            cur = []
    if cur:
        if groups:
            groups[-1] = np.concatenate([groups[-1]] + cur)
        else:
            return None
    # pass 1: plain median of log10(chi2) per group -> curve; pass 2: MAD sigma of the residuals from that curve
    # (the curve's own rise inside a wide bright group must not count as scatter)
    centres = [float(np.median(x[g])) for g in groups]
    med = [float(np.median(y[g])) for g in groups]
    nb = [int(len(g)) for g in groups]
    curve = Chi2Locus(centres=tuple(centres), med=tuple(med), scatter=tuple([1.0] * len(centres)), n=int(m.sum()),
                      source=source)
    resid = y - curve.predict(x)
    scat: list[float] = []
    for g in groups:
        r = resid[g]
        sb = MAD_TO_SIGMA * float(np.median(np.abs(r - np.median(r))))
        scat.append(sb if (math.isfinite(sb) and sb > 0) else float("nan"))
    good = [i for i, v in enumerate(scat) if math.isfinite(v)]
    centres = [centres[i] for i in good]
    med = [med[i] for i in good]
    scat = [scat[i] for i in good]
    nb = [nb[i] for i in good]
    if not centres:
        return None
    return Chi2Locus(
        centres=tuple(centres), med=tuple(med), scatter=tuple(scat), n=int(m.sum()), source=source, n_bin=tuple(nb)
    )


def frame_offset_locus(
    flux: Sequence[float] | np.ndarray,
    chi2: Sequence[float] | np.ndarray,
    *,
    night: Chi2Locus,
    converged: Sequence[bool] | np.ndarray | None = None,
    saturated: Sequence[bool] | np.ndarray | None = None,
) -> Chi2Locus:
    """Night curve shifted by the frame's plain median offset of log10(chi2); night curve if the frame is too small."""
    f = np.asarray(flux, dtype=np.float64)
    c = np.asarray(chi2, dtype=np.float64)
    m = _finite_mask(f, c, converged, saturated)
    n_f = int(m.sum())
    base = Chi2Locus(
        centres=night.centres, med=night.med, scatter=night.scatter, n=n_f, source="night", n_bin=night.n_bin
    )
    if n_f < MIN_N_LOCUS_FIT:
        return base
    lf = np.log10(f[m])
    off = float(np.median(np.log10(c[m]) - base.predict(lf)))
    return Chi2Locus(
        centres=night.centres, med=night.med, scatter=night.scatter, n=n_f, source="frame", offset=off, n_bin=night.n_bin
    )


def locus_residual_sigma(
    flux: float | np.ndarray,
    chi2: float | np.ndarray,
    locus: Chi2Locus,
) -> float | np.ndarray:
    """Residual of log10(chi2) from the reference at the star's flux, in units of that flux bin's sigma."""
    f = np.asarray(flux, dtype=np.float64)
    c = np.asarray(chi2, dtype=np.float64)
    scalar = np.ndim(f) == 0
    f = np.atleast_1d(f)
    c = np.atleast_1d(c)
    m = np.isfinite(f) & (f > 0) & np.isfinite(c) & (c > 0)
    out = np.full(f.shape, float("nan"), dtype=np.float64)
    lf = np.log10(f[m])
    out[m] = (np.log10(c[m]) - locus.predict(lf)) / locus.sigma(lf)
    return float(out[0]) if scalar else out


def chi2_ok_from_locus(
    *,
    converged: bool,
    chi2: float,
    resid_sigma: float,
    n_sigma: float = DEFAULT_NSIGMA,
) -> bool:
    """Unified SET: converged AND finite chi2 AND resid_sigma <= n_sigma.

    Nonfinite chi2 always fails (both iterative and grouped paths).
    """
    if not converged:
        return False
    if not math.isfinite(float(chi2)):
        return False
    if not math.isfinite(float(resid_sigma)):
        return False
    return float(resid_sigma) <= float(n_sigma)


def fit_night_locus_from_proc_dir(proc_dir: Path) -> Chi2Locus | None:
    """Pool locus members from all proc_*.csv under ``proc_dir``."""
    files = sorted(Path(proc_dir).glob("proc_*.csv"))
    if not files:
        return None
    fluxes: list[np.ndarray] = []
    chi2s: list[np.ndarray] = []
    convs: list[np.ndarray] = []
    sats: list[np.ndarray] = []
    usecols = [
        "psf_flux",
        "psf_chi2",
        "psf_converged",
        "psf_fit_ok",
        "likely_saturated",
        "is_saturated",
    ]
    for fp in files:
        hdr = pd.read_csv(fp, nrows=0).columns.tolist()
        cols = [c for c in usecols if c in hdr]
        if "psf_flux" not in cols or "psf_chi2" not in cols:
            continue
        df = read_vyvar_csv(fp, usecols=cols, low_memory=False)
        f = pd.to_numeric(df["psf_flux"], errors="coerce").to_numpy(dtype=np.float64)
        c = pd.to_numeric(df["psf_chi2"], errors="coerce").to_numpy(dtype=np.float64)
        if "psf_converged" in df.columns:
            cv = df["psf_converged"].fillna(False).astype(bool).to_numpy()
        else:
            cv = np.isfinite(f) & (f > 0) & np.isfinite(c)
        sat = np.zeros(len(df), dtype=bool)
        if "likely_saturated" in df.columns:
            sat |= df["likely_saturated"].fillna(False).astype(bool).to_numpy()
        if "is_saturated" in df.columns:
            sat |= df["is_saturated"].fillna(False).astype(bool).to_numpy()
        fluxes.append(f)
        chi2s.append(c)
        convs.append(cv)
        sats.append(sat)
    if not fluxes:
        return None
    return fit_chi2_locus(
        np.concatenate(fluxes),
        np.concatenate(chi2s),
        converged=np.concatenate(convs),
        saturated=np.concatenate(sats),
        source="night",
    )


def reapply_night_locus_to_proc_dir(
    proc_dir: Path,
    *,
    n_sigma: float | None = None,
    night_locus: Chi2Locus | None = None,
) -> dict[str, Any]:
    """Rewrite fit_ok against the night chi2 reference curve shifted by each frame's offset."""
    ns = float(n_sigma) if n_sigma is not None else DEFAULT_NSIGMA
    night = night_locus or fit_night_locus_from_proc_dir(proc_dir)
    summary: dict[str, Any] = {
        "n_sigma": ns,
        "night_locus": night.to_dict() if night else None,
        "frames": [],
        "n_rewritten": 0,
        "n_kept_frame": 0,
        "offsets": [],
    }
    if night is None:
        summary["error"] = "night_locus_unavailable"
        return summary
    files = sorted(Path(proc_dir).glob("proc_*.csv"))
    for fp in files:
        df = read_vyvar_csv(fp, low_memory=False)
        before = df.copy()
        if "psf_flux" not in df.columns or "psf_chi2" not in df.columns:
            continue
        flux = pd.to_numeric(df["psf_flux"], errors="coerce").to_numpy(dtype=np.float64)
        chi2 = pd.to_numeric(df["psf_chi2"], errors="coerce").to_numpy(dtype=np.float64)
        if "psf_converged" in df.columns:
            conv = df["psf_converged"].fillna(False).astype(bool).to_numpy()
        else:
            conv = np.isfinite(flux) & (flux > 0)
        sat = np.zeros(len(df), dtype=bool)
        if "likely_saturated" in df.columns:
            sat |= df["likely_saturated"].fillna(False).astype(bool).to_numpy()
        if "is_saturated" in df.columns:
            sat |= df["is_saturated"].fillna(False).astype(bool).to_numpy()
        locus = frame_offset_locus(flux, chi2, night=night, converged=conv, saturated=sat)
        source = locus.source
        resid = locus_residual_sigma(flux, chi2, locus)
        if not isinstance(resid, np.ndarray):
            resid = np.asarray([resid], dtype=np.float64)
        ok = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            ok[i] = chi2_ok_from_locus(
                converged=bool(conv[i]),
                chi2=float(chi2[i]) if math.isfinite(chi2[i]) else float("nan"),
                resid_sigma=float(resid[i]) if math.isfinite(resid[i]) else float("nan"),
                n_sigma=ns,
            )
        df["psf_converged"] = conv
        from proc_frame_store import round_proc_columns  # noqa: PLC0415

        df["psf_chi2_locus_resid_sigma"] = round_proc_columns(
            pd.DataFrame({"psf_chi2_locus_resid_sigma": resid})
        )["psf_chi2_locus_resid_sigma"].to_numpy()
        df["psf_fit_ok"] = ok
        # Only psf_* columns change; the guarded writer restores the file otherwise (INV-EXPORT-READ-ONLY-01).
        from psf_sidecar_guard import guarded_psf_sidecar_write  # noqa: PLC0415

        guarded_psf_sidecar_write(fp, df, before)
        entry = {
            "frame": fp.name,
            "source": source,
            "n_locus": int(locus.n),
            "offset": locus.offset,
            "n_ok": int(ok.sum()),
        }
        summary["frames"].append(entry)
        if source == "frame":
            summary["n_kept_frame"] += 1
            summary["offsets"].append(locus.offset)
        else:
            summary["n_rewritten"] += 1
    offs = summary["offsets"]
    if offs:
        arr = np.asarray(offs, dtype=np.float64)
        med_sig = float(np.median(night.scatter))
        summary["offset_spread"] = {
            "n": int(arr.size),
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "night_scatter_median": med_sig,
            "spread_over_night_scatter": float((np.max(arr) - np.min(arr)) / med_sig) if med_sig > 0 else float("nan"),
        }
    else:
        summary["offset_spread"] = {"n": 0}
    return summary


def _count_psf_fit_ok_in_sidecar(sidecar: Path) -> int:
    if not sidecar.is_file():
        return 0
    df = pd.read_csv(sidecar, usecols=["psf_fit_ok"], low_memory=False)
    col = df["psf_fit_ok"]
    if col.dtype == bool:
        return int(col.fillna(False).sum())
    s = col.astype(str).str.strip().str.lower()
    return int(s.isin(("true", "1", "yes")).sum())


def finalize_night_locus_for_inv_psf_frame_01(
    proc_dir: Path,
    frame_records: Sequence[dict[str, Any]] | None = None,
    *,
    n_sigma: float | None = None,
) -> dict[str, Any]:
    """Night-k + frame-a_f SET on all procs, then refresh ``n_ok`` for INV-PSF-FRAME-01.

    Per-frame PSF merge leaves ``psf_fit_ok`` closed until the night locus exists;
    this must run before ``finalize_epsf_frame_job``.
    """
    summary = reapply_night_locus_to_proc_dir(proc_dir, n_sigma=n_sigma)
    recs = [r for r in (frame_records or []) if isinstance(r, dict)]
    if recs:
        from proc_frame_store import proc_csv_path_for_aligned_fits

        root = Path(proc_dir)
        refreshed = 0
        for rec in recs:
            fn = str(rec.get("frame_name") or "")
            if not fn:
                continue
            sidecar = proc_csv_path_for_aligned_fits(root / fn)
            if sidecar.is_file():
                rec["n_ok"] = _count_psf_fit_ok_in_sidecar(sidecar)
                refreshed += 1
        summary["n_ok_refreshed_records"] = refreshed
    return summary


__all__ = [
    "Chi2Locus",
    "MIN_N_LOCUS_FIT",
    "DEFAULT_NSIGMA",
    "fit_chi2_locus",
    "frame_offset_locus",
    "MIN_N_BIN",
    "LOCUS_BIN_DEX",
    "locus_residual_sigma",
    "chi2_ok_from_locus",
    "fit_night_locus_from_proc_dir",
    "reapply_night_locus_to_proc_dir",
    "finalize_night_locus_for_inv_psf_frame_01",
]
