"""Sparse-field check-star trust - Howell 1988 triangulation + CI-based trust bands.

See docs/VYVAR_SPARSE_TRUST_SPEC.md. Citation: howellwarnockmitchell1988 (CITATIONS.bib).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from mag_constants import MAG_ERR_SCALE


@dataclass(frozen=True, slots=True)
class SparseTrustConfig:
    T_green: float = 1.5
    T_red: float = 4.0
    X2_RED: float = 0.0004  # (0.02 mag)^2
    stability_p_green: float = 0.01
    stability_p_red: float = 0.001


@dataclass(frozen=True, slots=True)
class TriangulationResult:
    sig2_K: float
    sig2_C1: float
    sig2_C2: float
    triangulation_clipped: bool


@dataclass(frozen=True, slots=True)
class ModelRatioCI:
    R: float
    R_lo: float
    R_hi: float
    v_obs: float
    v_model: float
    n_epochs: int


@dataclass(frozen=True, slots=True)
class CompStabilityResult:
    T: float
    p_value: float
    x2_pair_mag2: float
    s2_C1C2: float
    photon_denom: float


@dataclass(frozen=True, slots=True)
class SparseTrustStats:
    check_sparse: bool
    n_comps: int
    n_epochs: int
    trust_R: float
    trust_R_lo: float
    trust_R_hi: float
    comp_stability_p: float
    x2_pair_mag2: float
    triangulation_clipped: bool
    zp_sem_ratio: float | None = None
    single_comp: bool = False
    flags: tuple[str, ...] = field(default_factory=tuple)


def _finite_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = min(aa.size, bb.size)
    if n <= 0:
        return np.array([]), np.array([])
    ok = np.isfinite(aa[:n]) & np.isfinite(bb[:n])
    return aa[:n][ok], bb[:n][ok]


def sample_variance(series: np.ndarray) -> float:
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.var(x, ddof=1))


def diff_variance(m_a: np.ndarray, m_b: np.ndarray) -> float:
    a, b = _finite_pairs(m_a, m_b)
    if a.size < 2:
        return float("nan")
    return float(np.var(a - b, ddof=1))


def triangulate_variances(
    s2_KC1: float,
    s2_KC2: float,
    s2_C1C2: float,
) -> TriangulationResult:
    """Howell 1988 pairwise variance triangulation (Section 3.1)."""
    raw_k = (float(s2_KC1) + float(s2_KC2) - float(s2_C1C2)) / 2.0
    raw_c1 = (float(s2_KC1) + float(s2_C1C2) - float(s2_KC2)) / 2.0
    raw_c2 = (float(s2_KC2) + float(s2_C1C2) - float(s2_KC1)) / 2.0
    clipped = raw_k < 0 or raw_c1 < 0 or raw_c2 < 0
    return TriangulationResult(
        sig2_K=max(0.0, raw_k),
        sig2_C1=max(0.0, raw_c1),
        sig2_C2=max(0.0, raw_c2),
        triangulation_clipped=bool(clipped),
    )


def photon_corrected_excess(sig2_hat: float, pbar2: float) -> tuple[float, bool]:
    """Excess variance after photon subtraction (Section 3.2)."""
    raw = float(sig2_hat) - float(pbar2)
    return max(0.0, raw), bool(raw < 0)


def mean_pbar2(photon_mag: np.ndarray) -> float:
    p = np.asarray(photon_mag, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    return float(np.mean(p * p))


def sigma_zp_per_epoch(
    comp_fluxes: np.ndarray,
    photon_mag: np.ndarray,
    x2: np.ndarray,
) -> np.ndarray:
    """Sparse ZP noise per epoch (Section 3.3): sum w_i^2 * (p_i^2 + x2_i)."""
    flux = np.asarray(comp_fluxes, dtype=np.float64)
    phot = np.asarray(photon_mag, dtype=np.float64)
    x2_arr = np.asarray(x2, dtype=np.float64)
    n_comp, n_epochs = flux.shape
    out = np.full(n_epochs, float("nan"), dtype=np.float64)
    for t in range(n_epochs):
        f_t = flux[:, t]
        ok = np.isfinite(f_t) & (f_t > 0)
        if ok.sum() < 2:
            continue
        f_ok = f_t[ok]
        w = f_ok / float(np.sum(f_ok))
        p_ok = phot[ok, t]
        x_ok = x2_arr[ok]
        p2 = np.where(np.isfinite(p_ok), p_ok * p_ok, 0.0)
        x_ok = np.where(np.isfinite(x_ok), x_ok, 0.0)
        var = float(np.sum((w * w) * (p2 + x_ok)))
        out[t] = math.sqrt(max(var, 0.0))
    return out


def check_model_ratio_ci(v_obs: float, v_model: float, n_epochs: int) -> ModelRatioCI:
    """Variance ratio test with chi-square CI (Section 3.4)."""
    n = int(n_epochs)
    vm = float(v_model)
    vo = float(v_obs)
    if n < 2 or not (math.isfinite(vm) and vm > 0 and math.isfinite(vo) and vo >= 0):
        return ModelRatioCI(
            R=float("nan"),
            R_lo=float("nan"),
            R_hi=float("nan"),
            v_obs=vo,
            v_model=vm,
            n_epochs=n,
        )
    R = vo / vm
    df = n - 1
    chi_hi = float(stats.chi2.ppf(0.975, df))
    chi_lo = float(stats.chi2.ppf(0.025, df))
    return ModelRatioCI(
        R=R,
        R_lo=R * df / chi_hi if chi_hi > 0 else float("nan"),
        R_hi=R * df / chi_lo if chi_lo > 0 else float("nan"),
        v_obs=vo,
        v_model=vm,
        n_epochs=n,
    )


def comp_stability_test(
    s2_C1C2: float,
    photon_C1: np.ndarray,
    photon_C2: np.ndarray,
    *,
    sig2_C1_hat: float,
    sig2_C2_hat: float,
    pbar2_C1: float,
    pbar2_C2: float,
) -> CompStabilityResult:
    """Comp mutual stability F/chi2 test (Section 3.5)."""
    p1 = np.asarray(photon_C1, dtype=np.float64)
    p2 = np.asarray(photon_C2, dtype=np.float64)
    n = min(p1.size, p2.size)
    if n < 2:
        return CompStabilityResult(
            T=float("nan"),
            p_value=float("nan"),
            x2_pair_mag2=float("nan"),
            s2_C1C2=float(s2_C1C2),
            photon_denom=float("nan"),
        )
    pair_phot = p1[:n] * p1[:n] + p2[:n] * p2[:n]
    ok = np.isfinite(pair_phot)
    if ok.sum() < 2:
        denom = float("nan")
    else:
        denom = float(np.mean(pair_phot[ok]))
    s2 = float(s2_C1C2)
    if not (math.isfinite(denom) and denom > 0 and math.isfinite(s2)):
        T = float("nan")
        p_val = float("nan")
    else:
        T = (n - 1) * s2 / denom
        p_val = float(1.0 - stats.chi2.cdf(T, n - 1))
    x2_1, _ = photon_corrected_excess(sig2_C1_hat, pbar2_C1)
    x2_2, _ = photon_corrected_excess(sig2_C2_hat, pbar2_C2)
    x2_pair = max(x2_1, x2_2)
    return CompStabilityResult(
        T=T,
        p_value=p_val,
        x2_pair_mag2=x2_pair,
        s2_C1C2=s2,
        photon_denom=denom,
    )


def trust_band(
    *,
    R_hi: float,
    R_lo: float,
    stability_p: float,
    x2_pair_mag2: float,
    n_comps: int,
    triangulation_clipped: bool = False,
    cfg: SparseTrustConfig | None = None,
) -> tuple[str, tuple[str, ...]]:
    """CI-based trust band (Section 4). Returns (GREEN|YELLOW|RED, reason flags)."""
    th = cfg or SparseTrustConfig()
    flags: list[str] = []
    if int(n_comps) < 2:
        flags.append("single_comp")
        return "YELLOW", tuple(flags)
    if triangulation_clipped:
        flags.append("triangulation_clipped")
    p = float(stability_p)
    r_hi = float(R_hi)
    r_lo = float(R_lo)
    x2 = float(x2_pair_mag2)
    if math.isfinite(r_lo) and r_lo >= th.T_red:
        flags.append(f"R_lo>={th.T_red}")
        return "RED", tuple(flags)
    if math.isfinite(p) and p < th.stability_p_red and math.isfinite(x2) and x2 > th.X2_RED:
        flags.append("comp_pair_unstable")
        return "RED", tuple(flags)
    green = (
        math.isfinite(r_hi)
        and r_hi <= th.T_green
        and math.isfinite(p)
        and p >= th.stability_p_green
    )
    if green and not triangulation_clipped:
        return "GREEN", tuple(flags)
    if math.isfinite(p) and th.stability_p_red <= p < th.stability_p_green:
        flags.append("marginal_comp_stability")
    if math.isfinite(r_hi) and r_hi > th.T_green:
        flags.append("R_hi_exceeds_T_green")
    return "YELLOW", tuple(flags)


def _inst_to_flux(mag: np.ndarray) -> np.ndarray:
    m = np.asarray(mag, dtype=np.float64)
    out = np.full_like(m, float("nan"))
    ok = np.isfinite(m)
    out[ok] = 10.0 ** (-0.4 * m[ok])
    return out


def compute_sparse_trust_stats(
    *,
    kmag: np.ndarray,
    m_K: np.ndarray,
    comp_mags: dict[str, np.ndarray],
    comp_photon_mag: dict[str, np.ndarray],
    sigma_sys_mag: float,
    n_comps: int,
    cfg: SparseTrustConfig | None = None,
) -> SparseTrustStats:
    """End-to-end sparse trust statistics for one target night."""
    n_comp = int(n_comps)
    check_sparse = n_comp <= 2
    if n_comp < 1:
        return SparseTrustStats(
            check_sparse=True,
            n_comps=n_comp,
            n_epochs=0,
            trust_R=float("nan"),
            trust_R_lo=float("nan"),
            trust_R_hi=float("nan"),
            comp_stability_p=float("nan"),
            x2_pair_mag2=float("nan"),
            triangulation_clipped=False,
            single_comp=True,
        )

    comp_ids = list(comp_mags.keys())[: max(2, n_comp)]
    if len(comp_ids) < 2:
        return SparseTrustStats(
            check_sparse=True,
            n_comps=n_comp,
            n_epochs=0,
            trust_R=float("nan"),
            trust_R_lo=float("nan"),
            trust_R_hi=float("nan"),
            comp_stability_p=float("nan"),
            x2_pair_mag2=float("nan"),
            triangulation_clipped=False,
            single_comp=True,
        )

    c1, c2 = comp_ids[0], comp_ids[1]
    s2_kc1 = diff_variance(m_K, comp_mags[c1])
    s2_kc2 = diff_variance(m_K, comp_mags[c2])
    s2_c1c2 = diff_variance(comp_mags[c1], comp_mags[c2])
    tri = triangulate_variances(s2_kc1, s2_kc2, s2_c1c2)

    p_k = comp_photon_mag.get("__check__", np.full_like(m_K, float("nan")))
    p_c1 = comp_photon_mag.get(c1, np.full_like(m_K, float("nan")))
    p_c2 = comp_photon_mag.get(c2, np.full_like(m_K, float("nan")))
    pb_k = mean_pbar2(p_k)
    pb_c1 = mean_pbar2(p_c1)
    pb_c2 = mean_pbar2(p_c2)
    x2_k, _ = photon_corrected_excess(tri.sig2_K, pb_k)
    x2_c1, _ = photon_corrected_excess(tri.sig2_C1, pb_c1)
    x2_c2, _ = photon_corrected_excess(tri.sig2_C2, pb_c2)
    _ = x2_k  # used in model path via triangulation; retained for future per-star diagnostics

    comp_flux = np.vstack(
        [_inst_to_flux(comp_mags[c1]), _inst_to_flux(comp_mags[c2])],
    )
    phot_stack = np.vstack([p_c1, p_c2])
    x2_vec = np.array([x2_c1, x2_c2], dtype=np.float64)
    sigma_zp = sigma_zp_per_epoch(comp_flux, phot_stack, x2_vec)

    floor = float(sigma_sys_mag) if math.isfinite(float(sigma_sys_mag)) and float(sigma_sys_mag) > 0 else 0.0
    km = np.asarray(kmag, dtype=np.float64)
    pk = np.asarray(p_k, dtype=np.float64)
    n = min(km.size, pk.size, sigma_zp.size)
    sig_model = np.full(n, float("nan"), dtype=np.float64)
    for i in range(n):
        if not (math.isfinite(km[i]) and math.isfinite(sigma_zp[i]) and math.isfinite(pk[i])):
            continue
        v = pk[i] * pk[i] + sigma_zp[i] * sigma_zp[i] + floor * floor
        sig_model[i] = v
    v_model = float(np.nanmean(sig_model)) if np.isfinite(sig_model).any() else float("nan")
    v_obs = sample_variance(km)
    n_epochs = int(np.isfinite(km).sum())
    ratio = check_model_ratio_ci(v_obs, v_model, n_epochs)
    stab = comp_stability_test(
        s2_c1c2,
        p_c1,
        p_c2,
        sig2_C1_hat=tri.sig2_C1,
        sig2_C2_hat=tri.sig2_C2,
        pbar2_C1=pb_c1,
        pbar2_C2=pb_c2,
    )

    return SparseTrustStats(
        check_sparse=check_sparse,
        n_comps=n_comp,
        n_epochs=n_epochs,
        trust_R=ratio.R,
        trust_R_lo=ratio.R_lo,
        trust_R_hi=ratio.R_hi,
        comp_stability_p=stab.p_value,
        x2_pair_mag2=stab.x2_pair_mag2,
        triangulation_clipped=tri.triangulation_clipped,
    )


def sparse_trust_config_from_app(cfg: Any | None) -> SparseTrustConfig:
    if cfg is None:
        return SparseTrustConfig()
    return SparseTrustConfig(
        T_green=float(getattr(cfg, "sparse_trust_T_green", 1.5) or 1.5),
        T_red=float(getattr(cfg, "sparse_trust_T_red", 4.0) or 4.0),
        X2_RED=float(getattr(cfg, "sparse_trust_X2_RED", 0.0004) or 0.0004),
    )


def rel_flux_err_to_photon_mag(err_rel: np.ndarray) -> np.ndarray:
    """Convert relative flux err to mag-domain photon sigma."""
    e = np.asarray(err_rel, dtype=np.float64)
    out = np.full_like(e, float("nan"))
    ok = np.isfinite(e) & (e > 0)
    out[ok] = MAG_ERR_SCALE * e[ok]
    return out
