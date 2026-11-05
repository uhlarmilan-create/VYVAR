# -*- coding: ascii -*-
"""EPSF-CHI2-LOCUS-01: data-derived chi2(flux) reference for psf_fit_ok (CHI2-FLUX-REF-2026-10).

Sets ``psf_fit_ok`` of the frame-level PSF fit (RUN ePSF); no fixed chi2 threshold.

chi2 of a PSF fit is a noise term (about constant for faint stars) plus a PSF-model-mismatch term that grows
roughly with flux^2, so log10(chi2) against log10(flux) is curved, not a straight line (DAOPHOT handles the same
effect with a profile-error term, Stetson 1987). A single straight line, set by the many faint stars, put the
bright, well-measured stars 4-8 sigma above it and rejected them (Brno 534 g: 35-56 % of stars brighter than
log10 flux 4.5 rejected while their PSF/aperture ratio was as stable as that of the accepted ones).

Night reference: the night's stars are pooled and binned in log10(flux) (``LOCUS_BIN_DEX``; consecutive bins are
merged until a group holds ``MIN_N_BIN`` star x frame samples AND ``MIN_STARS_BIN`` distinct stars - G09-09: the
samples of one star share its persistent chi2 offset, so the scale's effective n is the number of stars); per group the plain median of log10(chi2) at the group's median log10(flux), and
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
from stats_core import MAD_TO_SIGMA, coerce_bool_series

from gaia_catalog_id import read_vyvar_csv

LOGGER = logging.getLogger(__name__)

# Statistical sample sizes (not equipment): a frame offset needs MIN_N_LOCUS_FIT stars, a bin statistic MIN_N_BIN
# samples from at least MIN_STARS_BIN distinct stars (G09-09, CHI2-LOCUS-STARS-2026-10: relative standard error of a
# MAD scale ~1.17 / sqrt(n) = 0.30 at n = 15 stars; Brno 534 g bright groups held 2-17 stars, their per-star chi2
# offsets 0.10-0.66 dex were as large as the pooled MAD).
MIN_N_LOCUS_FIT = 10
MIN_N_BIN = 30
MIN_STARS_BIN = 15
# Flux resolution of the reference: 0.25 dex = 0.625 mag.
LOCUS_BIN_DEX = 0.25
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
    n_stars_bin: tuple[int, ...] = field(default=())
    note: str = ""

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
            "n_stars_bin": [int(v) for v in self.n_stars_bin],
            "min_stars_bin": MIN_STARS_BIN,
            "note": self.note,
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
    star_ids: Sequence[Any] | np.ndarray,
    converged: Sequence[bool] | np.ndarray | None = None,
    saturated: Sequence[bool] | np.ndarray | None = None,
    source: str = "night",
) -> Chi2Locus | None:
    """Binned reference curve: plain median and MAD sigma of log10(chi2) per log10(flux) bin.

    ``star_ids`` names the star of every sample (a group needs ``MIN_STARS_BIN`` distinct stars). Returns None if
    the night has fewer than ``MIN_N_BIN`` samples. A night with fewer than ``MIN_STARS_BIN`` stars in all is one
    group (WARNING, ``note`` in the locus record).
    """
    f = np.asarray(flux, dtype=np.float64)
    c = np.asarray(chi2, dtype=np.float64)
    sid = np.asarray([str(v) for v in star_ids], dtype=object)
    if sid.size != f.size:
        raise ValueError(f"fit_chi2_locus: {sid.size} star ids for {f.size} samples")
    m = _finite_mask(f, c, converged, saturated)
    if int(m.sum()) < MIN_N_BIN:
        return None
    x = np.log10(f[m])
    y = np.log10(c[m])
    sid = sid[m]
    lo = math.floor(float(x.min()) / LOCUS_BIN_DEX) * LOCUS_BIN_DEX
    idx = np.floor((x - lo) / LOCUS_BIN_DEX).astype(int)
    # consecutive LOCUS_BIN_DEX bins are merged until a group holds MIN_N_BIN samples AND MIN_STARS_BIN distinct stars
    # (sparse bright end); a sparse remainder joins the previous group
    groups: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    note = ""
    for b in np.unique(idx):
        cur.append(np.flatnonzero(idx == b))
        g = np.concatenate(cur)
        if g.size >= MIN_N_BIN and np.unique(sid[g]).size >= MIN_STARS_BIN:
            groups.append(g)
            cur = []
    if cur:
        if groups:
            groups[-1] = np.concatenate([groups[-1]] + cur)
        else:
            groups = [np.concatenate(cur)]
            note = f"only {np.unique(sid).size} stars (< MIN_STARS_BIN {MIN_STARS_BIN}): one flux group"
            LOGGER.warning("[ePSF] chi2 locus: %s", note)
    # pass 1: plain median of log10(chi2) per group -> curve; pass 2: MAD sigma of the residuals from that curve
    # (the curve's own rise inside a wide bright group must not count as scatter)
    centres = [float(np.median(x[g])) for g in groups]
    med = [float(np.median(y[g])) for g in groups]
    nb = [int(len(g)) for g in groups]
    ns = [int(np.unique(sid[g]).size) for g in groups]
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
    ns = [ns[i] for i in good]
    if not centres:
        return None
    return Chi2Locus(
        centres=tuple(centres), med=tuple(med), scatter=tuple(scat), n=int(m.sum()), source=source, n_bin=tuple(nb),
        n_stars_bin=tuple(ns), note=note,
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
        centres=night.centres, med=night.med, scatter=night.scatter, n=n_f, source="night", n_bin=night.n_bin,
        n_stars_bin=night.n_stars_bin, note=night.note,
    )
    if n_f < MIN_N_LOCUS_FIT:
        return base
    lf = np.log10(f[m])
    off = float(np.median(np.log10(c[m]) - base.predict(lf)))
    return Chi2Locus(
        centres=night.centres, med=night.med, scatter=night.scatter, n=n_f, source="frame", offset=off, n_bin=night.n_bin,
        n_stars_bin=night.n_stars_bin, note=night.note,
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


#: The proc columns the night locus fit reads.
LOCUS_PROC_COLUMNS: tuple[str, ...] = (
    "catalog_id",
    "psf_flux",
    "psf_chi2",
    "psf_converged",
    "psf_fit_ok",
    "likely_saturated",
    "is_saturated",
)


def locus_columns_of(df: pd.DataFrame) -> pd.DataFrame:
    """The ``LOCUS_PROC_COLUMNS`` of a proc table read in full, in file order: the same table as a read of the
    file with those columns only (G09-27: the ePSF merge keeps it from its guarded write's read-back)."""
    return df[[c for c in df.columns if c in LOCUS_PROC_COLUMNS]]


def locus_table_key(path: Path | str) -> str:
    """Key of a proc file in a ``tables`` mapping: its resolved path (the ePSF job lists resolved FITS paths,
    the locus pass globs the frames root as given)."""
    return str(Path(path).resolve())


def fit_night_locus_from_proc_dir(
    proc_dir: Path,
    tables: dict[str, pd.DataFrame] | None = None,
) -> Chi2Locus | None:
    """Pool locus members from all proc_*.csv under ``proc_dir``.

    ``tables``: ``{locus_table_key(proc path): locus_columns_of(table)}`` of files whose current content the caller
    already read (G09-27); every other file is read here."""
    files = sorted(Path(proc_dir).glob("proc_*.csv"))
    if not files:
        return None
    fluxes: list[np.ndarray] = []
    chi2s: list[np.ndarray] = []
    convs: list[np.ndarray] = []
    sats: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for fp in files:
        df = tables.get(locus_table_key(fp)) if tables else None
        if df is None:
            # G09-27: one read (no nrows=0 header read); the same columns in file order
            df = read_vyvar_csv(fp, usecols=lambda c: c in LOCUS_PROC_COLUMNS, low_memory=False)
        if "psf_flux" not in df.columns or "psf_chi2" not in df.columns:
            continue
        if "catalog_id" not in df.columns:
            raise ValueError(f"{fp.name}: no catalog_id - the chi2 locus counts distinct stars per flux bin")
        f = pd.to_numeric(df["psf_flux"], errors="coerce").to_numpy(dtype=np.float64)
        c = pd.to_numeric(df["psf_chi2"], errors="coerce").to_numpy(dtype=np.float64)
        if "psf_converged" in df.columns:
            cv = coerce_bool_series(df["psf_converged"]).to_numpy()
        else:
            cv = np.isfinite(f) & (f > 0) & np.isfinite(c)
        sat = np.zeros(len(df), dtype=bool)
        if "likely_saturated" in df.columns:
            sat |= coerce_bool_series(df["likely_saturated"]).to_numpy()
        if "is_saturated" in df.columns:
            sat |= coerce_bool_series(df["is_saturated"]).to_numpy()
        fluxes.append(f)
        chi2s.append(c)
        convs.append(cv)
        sats.append(sat)
        ids.append(df["catalog_id"].astype(str).to_numpy())
    if not fluxes:
        return None
    return fit_chi2_locus(
        np.concatenate(fluxes),
        np.concatenate(chi2s),
        star_ids=np.concatenate(ids),
        converged=np.concatenate(convs),
        saturated=np.concatenate(sats),
        source="night",
    )


def reapply_night_locus_to_proc_dir(
    proc_dir: Path,
    *,
    n_sigma: float | None = None,
    night_locus: Chi2Locus | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Rewrite fit_ok against the night chi2 reference curve shifted by each frame's offset.

    ``tables``: already-read locus columns for the night locus fit (``fit_night_locus_from_proc_dir``)."""
    return _reapply_night_locus(proc_dir, n_sigma=n_sigma, night_locus=night_locus, tables=tables)[0]


def _reapply_night_locus(
    proc_dir: Path,
    *,
    n_sigma: float | None,
    night_locus: Chi2Locus | None,
    tables: dict[str, pd.DataFrame] | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """``reapply_night_locus_to_proc_dir`` plus ``{str(proc path): number of psf_fit_ok rows written}``."""
    ns = float(n_sigma) if n_sigma is not None else DEFAULT_NSIGMA
    night = night_locus or fit_night_locus_from_proc_dir(proc_dir, tables)
    summary: dict[str, Any] = {
        "n_sigma": ns,
        "night_locus": night.to_dict() if night else None,
        "frames": [],
        "n_rewritten": 0,
        "n_kept_frame": 0,
        "offsets": [],
    }
    n_ok_written: dict[str, int] = {}
    if night is None:
        summary["error"] = "night_locus_unavailable"
        return summary, n_ok_written
    files = sorted(Path(proc_dir).glob("proc_*.csv"))
    for fp in files:
        df = read_vyvar_csv(fp, low_memory=False)
        before = df.copy()
        if "psf_flux" not in df.columns or "psf_chi2" not in df.columns:
            continue
        flux = pd.to_numeric(df["psf_flux"], errors="coerce").to_numpy(dtype=np.float64)
        chi2 = pd.to_numeric(df["psf_chi2"], errors="coerce").to_numpy(dtype=np.float64)
        if "psf_converged" in df.columns:
            conv = coerce_bool_series(df["psf_converged"]).to_numpy()
        else:
            conv = np.isfinite(flux) & (flux > 0)
        sat = np.zeros(len(df), dtype=bool)
        if "likely_saturated" in df.columns:
            sat |= coerce_bool_series(df["likely_saturated"]).to_numpy()
        if "is_saturated" in df.columns:
            sat |= coerce_bool_series(df["is_saturated"]).to_numpy()
        locus = frame_offset_locus(flux, chi2, night=night, converged=conv, saturated=sat)
        source = locus.source
        resid = locus_residual_sigma(flux, chi2, locus)
        if not isinstance(resid, np.ndarray):
            resid = np.asarray([resid], dtype=np.float64)
        # G09-29: chi2_ok_from_locus for every row at once (converged, finite chi2, resid <= n_sigma)
        ok = np.asarray(conv, dtype=bool) & np.isfinite(chi2) & np.isfinite(resid) & (resid <= float(ns))
        df["psf_converged"] = conv
        from proc_frame_store import round_proc_columns  # noqa: PLC0415

        df["psf_chi2_locus_resid_sigma"] = round_proc_columns(
            pd.DataFrame({"psf_chi2_locus_resid_sigma": resid})
        )["psf_chi2_locus_resid_sigma"].to_numpy()
        df["psf_fit_ok"] = ok
        # Only psf_* columns change; the guarded writer restores the file otherwise (INV-EXPORT-READ-ONLY-01).
        from psf_sidecar_guard import guarded_psf_sidecar_write  # noqa: PLC0415

        guarded_psf_sidecar_write(fp, df, before)
        n_ok_written[str(fp)] = int(ok.sum())
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
    return summary, n_ok_written


def _count_psf_fit_ok_in_sidecar(sidecar: Path) -> int:
    if not sidecar.is_file():
        return 0
    df = pd.read_csv(sidecar, usecols=["psf_fit_ok"], low_memory=False)
    return int(coerce_bool_series(df["psf_fit_ok"]).sum())


def finalize_night_locus_for_inv_psf_frame_01(
    proc_dir: Path,
    frame_records: Sequence[dict[str, Any]] | None = None,
    *,
    n_sigma: float | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Night-k + frame-a_f SET on all procs, then refresh ``n_ok`` for INV-PSF-FRAME-01.

    Per-frame PSF merge leaves ``psf_fit_ok`` closed until the night locus exists;
    this must run before ``finalize_epsf_frame_job``. ``tables``: see ``fit_night_locus_from_proc_dir``.
    """
    summary, n_ok_written = _reapply_night_locus(proc_dir, n_sigma=n_sigma, night_locus=None, tables=tables)
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
                # G09-27: the psf_fit_ok count the locus pass just wrote to this file; a file it did not
                # rewrite (no PSF columns) is counted from disk
                n_ok = n_ok_written.get(str(sidecar))
                rec["n_ok"] = n_ok if n_ok is not None else _count_psf_fit_ok_in_sidecar(sidecar)
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
    "MIN_STARS_BIN",
    "LOCUS_BIN_DEX",
    "locus_residual_sigma",
    "chi2_ok_from_locus",
    "fit_night_locus_from_proc_dir",
    "LOCUS_PROC_COLUMNS",
    "locus_columns_of",
    "locus_table_key",
    "reapply_night_locus_to_proc_dir",
    "finalize_night_locus_for_inv_psf_frame_01",
]
