"""
Download and analyze TESS cutout light curves for variable-star candidates (lightkurve).
"""
from __future__ import annotations

import json
import logging
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.timeseries import LombScargle

logger = logging.getLogger(__name__)

_MPL_LOCK = threading.Lock()

# Lomb-Scargle multi-band (dni); od ~28 min do 100 dní
_LIST_SECTION: list[float] = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
_PERIOD_MIN_SEARCH_D = float(_LIST_SECTION[0])
_PERIOD_MAX_SEARCH_D = float(_LIST_SECTION[-1])


def _get_optimal_window(period_days: float | None) -> int:
    """``window_length`` pre ``flatten()`` alebo ``-1`` = preskočiť flatten (dlhé periódy).

    - P < 0.1 d (~2.4 h): 51 (ultrakrátké periódy).
    - 0.1 d ≤ P ≤ 15 d: 201.
    - P > 15 d (Mira atď.): ``-1`` — len ``normalize()``, zachovať pomalý trend.
    - Neznáme / neplatné P: predvolene 201.
    """
    if period_days is None:
        return 201
    try:
        p = float(period_days)
    except (TypeError, ValueError):
        return 201
    if not np.isfinite(p) or p <= 0:
        return 201
    if p < 0.1:
        return 51
    if p > 15.0:
        return -1
    return 201


def _dynamic_window_length(lc: Any, min_window: int = 301) -> int:
    """
    Dynamický window_length pre flatten — vždy 10% dĺžky LC, minimum 301.
    Kratší okno by mohol odstrániť signal pri krátkych periódach (P < 0.5 d).
    Výsledok musí byť odd (lightkurve požiadavka).
    """
    try:
        n = int(len(lc))
    except Exception:  # noqa: BLE001
        return min_window
    w = max(min_window, int(0.10 * n))
    return w if w % 2 == 1 else w + 1


def _prepare_lc_for_period(
    lc: Any,
    *,
    window_length: int | None,
    use_flatten: bool,
    break_tolerance: int = 50,
) -> Any:
    """
    Pripraví LC pre period search:
    - normalize na medián=1 (vždy)
    - flatten so Savitzky-Golay (len ak use_flatten=True)

    window_length=None → dynamický výpočet (10% dĺžky LC, min 301).
    break_tolerance=50 odporúčaný pre TESS (default lightkurve=5 overfituje).
    """
    try:
        lc = lc.normalize()
    except Exception as exc:  # noqa: BLE001
        logger.warning("normalize skipped: %s", exc)

    if not use_flatten:
        return lc

    wl = (
        _dynamic_window_length(lc)
        if (window_length is None or int(window_length) < 3)
        else int(window_length)
    )
    if wl % 2 == 0:
        wl += 1

    try:
        logger.debug(
            "[TESS flatten] n_points=%d window_length=%d break_tolerance=%d",
            int(len(lc)),
            wl,
            break_tolerance,
        )
        return lc.flatten(
            window_length=wl,
            break_tolerance=break_tolerance,
            niters=3,
            sigma=3,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("flatten skipped in _prepare_lc_for_period: %s", exc)
        return lc


def _lomb_scargle_best_period(
    lc: Any,
    *,
    window_length: int = 101,
    use_flatten: bool = True,
) -> float | None:
    """Lomb-Scargle cez ``_LIST_SECTION``; voliteľný ``flatten`` pred periodogramom."""
    work = _prepare_lc_for_period(lc, window_length=None, use_flatten=use_flatten)
    best_pg = None
    best_power = -1.0
    for i in range(len(_LIST_SECTION) - 1):
        p0, p1 = float(_LIST_SECTION[i]), float(_LIST_SECTION[i + 1])
        if not (np.isfinite(p0) and np.isfinite(p1) and p1 > p0):
            continue
        try:
            pg = work.to_periodogram(
                minimum_period=p0 * u.day,
                maximum_period=p1 * u.day,
                oversample_factor=500,
            )
            pw = np.asarray(pg.power, dtype=float)
            if pw.size == 0:
                continue
            mx = float(np.nanmax(pw))
            if mx > best_power:
                best_power = mx
                best_pg = pg
        except Exception as exc:  # noqa: BLE001
            logger.debug("periodogram section %s-%s skipped: %s", p0, p1, exc)
            continue
    if best_pg is None:
        return None
    try:
        return float(best_pg.period_at_max_power.value)
    except Exception:
        return None


def _lc_to_ty_arrays(lc: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Čas (d), flux, voliteľné chyby pre ``astropy.timeseries.LombScargle``."""
    t = np.asarray(lc.time.value if hasattr(lc.time, "value") else lc.time, dtype=float)
    y = np.asarray(lc.flux.value if hasattr(lc.flux, "value") else lc.flux, dtype=float)
    dy: np.ndarray | None = None
    try:
        fe = getattr(lc, "flux_err", None)
        if fe is not None:
            dy = np.asarray(fe.value if hasattr(fe, "value") else fe, dtype=float)
    except Exception:  # noqa: BLE001
        dy = None
    m = np.isfinite(t) & np.isfinite(y)
    if dy is not None:
        m &= np.isfinite(dy) & (dy > 0)
    t, y = t[m], y[m]
    if dy is not None:
        dy = dy[m]
    if y.size:
        y = y - float(np.nanmedian(y))
    return t, y, dy


def _iterative_sigma_clip_lc(
    lc: Any,
    *,
    sigma: float = 3.0,
    max_iter: int = 8,
    min_points: int = 10,
    min_frac_keep: float = 0.25,
) -> Any:
    """Iteratívne 3σ orezanie fluxu (Peranso-style) na odstránenie CR / dump artefaktov."""
    work = lc
    n0 = int(len(lc)) if hasattr(lc, "__len__") else 0
    for _ in range(int(max_iter)):
        f = np.asarray(work.flux.value if hasattr(work.flux, "value") else work.flux, dtype=float)
        n = int(np.sum(np.isfinite(f)))
        if n < int(min_points):
            break
        med = float(np.nanmedian(f))
        mad = float(np.nanmedian(np.abs(f - med)))
        scale = 1.4826 * mad if mad > 0 and np.isfinite(mad) else float(np.nanstd(f))
        if not np.isfinite(scale) or scale <= 0:
            break
        lo, hi = med - float(sigma) * scale, med + float(sigma) * scale
        keep = np.isfinite(f) & (f >= lo) & (f <= hi)
        if int(np.sum(keep)) == len(f):
            break
        if int(np.sum(keep)) < max(int(min_points), int(min_frac_keep * max(n0, n))):
            break
        work = work[keep]
    return work


def _find_period_anova(
    lc: Any,
    window_length: int = 101,
    *,
    use_flatten: bool = True,
) -> float | None:
    """
    „ANOVA“ stopa: Lomb–Scargle v ``astropy.timeseries.LombScargle`` po rovnakých pásoch ako L-S
    v lightkurve, nezávislé na ``to_periodogram`` — krížová validácia s PDM/BLS.
    """
    work = _prepare_lc_for_period(lc, window_length=None, use_flatten=use_flatten)
    t, y, dy = _lc_to_ty_arrays(work)
    if t.size < 10:
        return None
    try:
        ls = LombScargle(t, y, dy=dy, normalization="standard")
    except Exception as exc:  # noqa: BLE001
        logger.debug("[TESS ANOVA/LS] LombScargle init failed: %s", exc)
        return None
    best_p: float | None = None
    best_power = -1.0
    for i in range(len(_LIST_SECTION) - 1):
        p0, p1 = float(_LIST_SECTION[i]), float(_LIST_SECTION[i + 1])
        if not (np.isfinite(p0) and np.isfinite(p1) and p1 > p0):
            continue
        try:
            frequency = np.geomspace(1.0 / p1, 1.0 / p0, num=2048)
            power = np.asarray(ls.power(frequency), dtype=float)
            if power.size == 0:
                continue
            j = int(np.nanargmax(power))
            pw = float(power[j])
            if pw > best_power:
                best_power = pw
                best_p = float(1.0 / float(frequency[j]))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[TESS ANOVA/LS] band %s-%s skipped: %s", p0, p1, exc)
            continue
    return best_p


def _lomb_standard_power_at_frequency(lc: Any, frequency_per_day: float) -> float | None:
    """Výkon Lomb–Scargle (standard) pri danej frekvencii [1/deň]."""
    t, y, dy = _lc_to_ty_arrays(lc)
    if t.size < 10 or not (np.isfinite(frequency_per_day) and frequency_per_day > 0):
        return None
    try:
        ls = LombScargle(t, y, dy=dy, normalization="standard")
        return float(ls.power(float(frequency_per_day)))
    except Exception:  # noqa: BLE001
        return None


def _harmonic_refine_period(
    lc: Any,
    p_candidate: float | None,
    *,
    power_ratio_threshold: float = 1.35,
) -> tuple[float | None, str]:
    """
    Porovná výkon pri P, 0.5P a 2P; ak je 2P výrazne silnejšie (typické EB s aliasom P/2), preferuj 2P.
    Ak dominuje 0.5P a je v rozsahu hľadania, preferuj P/2.
    """
    if p_candidate is None:
        return None, ""
    p = float(p_candidate)
    if not (np.isfinite(p) and p > 0):
        return p_candidate, ""
    f = 1.0 / p
    pwr_p = _lomb_standard_power_at_frequency(lc, f)
    pwr_2p = _lomb_standard_power_at_frequency(lc, 0.5 * f)
    pwr_half = _lomb_standard_power_at_frequency(lc, 2.0 * f)
    if pwr_p is None:
        return p_candidate, ""
    thr = float(power_ratio_threshold)
    p2 = 2.0 * p
    ph = 0.5 * p
    if (
        pwr_2p is not None
        and pwr_2p > thr * float(pwr_p)
        and np.isfinite(p2)
        and _PERIOD_MIN_SEARCH_D <= p2 <= _PERIOD_MAX_SEARCH_D
    ):
        return float(p2), "harmonic(2P preferred)"
    if (
        pwr_half is not None
        and pwr_half > thr * float(pwr_p)
        and np.isfinite(ph)
        and _PERIOD_MIN_SEARCH_D <= ph <= _PERIOD_MAX_SEARCH_D
    ):
        return float(ph), "harmonic(0.5P preferred)"
    return p_candidate, ""


@dataclass
class TessSectorResult:
    sector: int
    jd_start: float
    jd_end: float
    period_found: float | None = None
    period_pdm: float | None = None
    period_bls: float | None = None
    period_anova: float | None = None
    period_consensus: float | None = None
    # "lomb-scargle" | "pdm" | "bls" | "consensus(...)" | "+anova" | "|harmonic(...)"
    period_method_used: str = "lomb-scargle"
    period_2p: float | None = None
    harmonic_note: str | None = None
    n_points_before_sigma_clip: int = 0
    lc_raw_path: str | None = None
    plot_raw_path: str | None = None
    plot_phased_p_path: str | None = None
    plot_phased_2p_path: str | None = None
    blend_check_path: str | None = None
    n_points: int = 0
    amplitude_ppt: float | None = None
    snr: float | None = None
    flux_std: float | None = None
    error: str | None = None


@dataclass
class TessResult:
    catalog_id: str
    ra: float
    dec: float
    mag: float | None
    sectors: list[TessSectorResult] = field(default_factory=list)
    period_consensus: float | None = None
    period_anova_consensus: float | None = None
    period_2p_consensus: float | None = None
    output_dir: str = ""
    total_sectors_found: int = 0
    total_sectors_ok: int = 0
    error_global: str | None = None
    period_reliability: str = "unknown"
    period_reliability_reason: str = ""

    def has_data(self) -> bool:
        return any((s.error is None and s.n_points > 0) for s in self.sectors)

    def best_period(self) -> float | None:
        return self.period_consensus

    def summary_text(self) -> str:
        if self.error_global:
            return f"TESS: chyba — {self.error_global}"
        if not self.has_data():
            return "TESS: žiadne platné sektory"
        p = self.period_consensus
        pa = self.period_anova_consensus
        p2 = self.period_2p_consensus
        ps = f"P={p:.6g} d" if p is not None else "P=—"
        pas = f"P_anova={pa:.6g} d" if pa is not None else "P_anova=—"
        p2s = f"2P={p2:.6g} d" if p2 is not None else "2P=—"
        return (
            f"TESS: {self.total_sectors_ok}/{self.total_sectors_found} sektorov OK | {ps} | {pas} | {p2s} | {self.output_dir}"
        )


def _get_aperture_params(mag: float | None) -> tuple[int, int, int, int, int]:
    if mag is None or not np.isfinite(mag):
        return (4, 4, 2, 2, 10)
    if mag < 8:
        return (7, 7, 7, 7, 20)
    if mag < 10:
        return (8, 8, 5, 5, 20)
    if mag < 13:
        return (6, 6, 3, 3, 14)
    if mag < 15.5:
        return (4, 4, 2, 2, 10)
    return (5, 5, 1, 1, 10)


def _delete_error(lc: Any, start: int = 40, end: int = 40, center: int = 160) -> Any:
    n = int(len(lc))
    if n == 0:
        return lc
    keep = np.ones(n, dtype=bool)
    if n > start:
        keep[:start] = False
    if n > end:
        keep[-end:] = False
    mid = n // 2
    half = center // 2
    c0 = max(0, mid - half)
    c1 = min(n, mid + half)
    if c1 > c0:
        keep[c0:c1] = False
    if not np.any(keep):
        return lc
    return lc[keep]


def _find_period(
    lc: Any,
    window_length: int = 101,
    period_hint: float | None = None,
    *,
    ignore_period_hint: bool = False,
    use_flatten_for_ls: bool = True,
) -> float | None:
    """Lomb-Scargle; ``period_hint`` preskočí hľadanie, ak je daný a ``ignore_period_hint`` je False.

    Ak je ``lc`` už po ``flatten()`` / ``normalize()``, nastav ``use_flatten_for_ls=False``,
    aby sa nevolal druhý ``flatten`` pred periodogramom.
    """
    if (
        not ignore_period_hint
        and period_hint is not None
        and np.isfinite(period_hint)
        and float(period_hint) > 0
    ):
        return float(period_hint)
    return _lomb_scargle_best_period(
        lc,
        window_length=int(window_length),
        use_flatten=bool(use_flatten_for_ls),
    )


def _find_period_pdm(
    lc: Any,
    window_length: int = 101,
    *,
    use_flatten: bool = True,
    minimum_period_days: float = _PERIOD_MIN_SEARCH_D,
    maximum_period_days: float = _PERIOD_MAX_SEARCH_D,
    n_periods: int = 1000,
    n_bins: int = 10,
) -> float | None:
    """
    Phase Dispersion Minimization — vlastná numpy implementácia.
    lightkurve 2.6.0 nepodporuje method='pdm'.
    """
    try:
        work = _prepare_lc_for_period(
            lc, window_length=None, use_flatten=use_flatten
        )
        time = np.asarray(work.time.value, dtype=float)
        flux = np.asarray(work.flux.value, dtype=float)

        var_total = np.var(flux)
        if var_total == 0 or len(time) < 20:
            return None

        periods = np.linspace(
            float(minimum_period_days),
            float(maximum_period_days),
            int(n_periods),
        )
        best_p, best_theta = None, np.inf

        for p in periods:
            phase = (time % p) / p
            idx = np.argsort(phase)
            flux_s = flux[idx]
            bins = np.array_split(flux_s, n_bins)
            weighted_var = sum(len(b) * np.var(b) for b in bins if len(b) > 1)
            n_total = sum(len(b) for b in bins if len(b) > 1)
            if n_total == 0:
                continue
            theta = (weighted_var / n_total) / var_total
            if theta < best_theta:
                best_theta = theta
                best_p = p

        return float(best_p) if best_p is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TESS PDM] failed: %s", exc)
        return None


def _find_period_bls(
    lc: Any,
    window_length: int = 101,
    *,
    use_flatten: bool = True,
    minimum_period_days: float = _PERIOD_MIN_SEARCH_D,
    maximum_period_days: float = _PERIOD_MAX_SEARCH_D,
) -> float | None:
    try:
        work = _prepare_lc_for_period(
            lc, window_length=None, use_flatten=use_flatten
        )
        # BLS vyžaduje duration < minimum_period
        # Použiť zlomky min_p, nie absolútne hodnoty
        min_p = max(float(minimum_period_days), 0.05)  # BLS nestabilné pod 0.05 d
        durations = [min_p * k for k in (0.05, 0.10, 0.20) if min_p * k < min_p]
        if not durations:
            durations = [min_p * 0.1]
        pg = work.to_periodogram(
            method="bls",
            minimum_period=min_p,  # float, bez u.day
            maximum_period=float(maximum_period_days),
            frequency_factor=15,
            duration=durations,
        )
        p = pg.period_at_max_power
        # Bezpečné čítanie — astropy 7.x Quantity
        if hasattr(p, "to_value"):
            return float(p.to_value("d"))
        return float(p.value)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TESS BLS] failed: %s", exc)
        return None


def _period_consensus(
    p_ls: float | None,
    p_pdm: float | None,
    p_bls: float | None,
    p_anova: float | None = None,
    tolerance: float = 0.05,
    *,
    short_period_threshold: float = 0.15,
    short_tol_pdm_bls: float = 0.12,
) -> tuple[float | None, str]:
    """
    Konsenzná perióda (2+ zhody); voliteľná štvrtá stopa ``p_anova`` (astropy Lomb–Scargle).

    Pri veľmi krátkych periódách (< short_period_threshold d) sa PDM+BLS zhoda
    posudzuje miernejším ``short_tol_pdm_bls``.

    Priorita (normálne): L-S+PDM > L-S+BLS > PDM+BLS (+ vetvy s anova)
    Priorita (krátke P): PDM+BLS (miernejší prah) > …
    Ak žiadna zhoda → L-S alebo ``None`` s method="lomb-scargle".
    """

    def agree(a: float | None, b: float | None, tol: float) -> bool:
        if a is None or b is None:
            return False
        try:
            af = float(a)
            bf = float(b)
        except Exception:  # noqa: BLE001
            return False
        if not (np.isfinite(af) and np.isfinite(bf) and af > 0 and bf > 0):
            return False
        return abs(af - bf) / max(af, bf) <= float(tol)

    vals = [float(x) for x in (p_ls, p_pdm, p_bls) if x is not None and np.isfinite(float(x)) and float(x) > 0]
    is_short = bool(vals) and min(vals) < float(short_period_threshold)

    p_base: float | None
    m_base: str

    if is_short:
        if agree(p_pdm, p_bls, short_tol_pdm_bls):
            p_base, m_base = (float(p_pdm) + float(p_bls)) / 2.0, "consensus(pdm+bls)"
        elif agree(p_ls, p_pdm, tolerance):
            p_base, m_base = (float(p_ls) + float(p_pdm)) / 2.0, "consensus(ls+pdm)"
        elif agree(p_ls, p_bls, tolerance):
            p_base, m_base = (float(p_ls) + float(p_bls)) / 2.0, "consensus(ls+bls)"
        elif agree(p_pdm, p_bls, tolerance):
            p_base, m_base = (float(p_pdm) + float(p_bls)) / 2.0, "consensus(pdm+bls)"
        else:
            p_base = float(p_ls) if p_ls is not None else None
            m_base = "lomb-scargle"
    else:
        if agree(p_ls, p_pdm, tolerance):
            p_base, m_base = (float(p_ls) + float(p_pdm)) / 2.0, "consensus(ls+pdm)"
        elif agree(p_ls, p_bls, tolerance):
            p_base, m_base = (float(p_ls) + float(p_bls)) / 2.0, "consensus(ls+bls)"
        elif agree(p_pdm, p_bls, tolerance):
            p_base, m_base = (float(p_pdm) + float(p_bls)) / 2.0, "consensus(pdm+bls)"
        else:
            p_base = float(p_ls) if p_ls is not None else None
            m_base = "lomb-scargle"

    if p_anova is None or not (np.isfinite(float(p_anova)) and float(p_anova) > 0):
        return p_base, m_base
    pa = float(p_anova)

    if p_base is not None and agree(p_base, pa, tolerance):
        return (float(p_base) + pa) / 2.0, f"{m_base}+anova"

    if agree(p_ls, pa, tolerance) and agree(p_pdm, pa, tolerance):
        return (float(p_ls) + float(p_pdm) + pa) / 3.0, "consensus(ls+pdm+anova)"
    if agree(p_ls, pa, tolerance) and agree(p_bls, pa, tolerance):
        return (float(p_ls) + float(p_bls) + pa) / 3.0, "consensus(ls+bls+anova)"
    if agree(p_pdm, pa, tolerance) and agree(p_bls, pa, tolerance):
        return (float(p_pdm) + float(p_bls) + pa) / 3.0, "consensus(pdm+bls+anova)"
    if agree(p_ls, pa, tolerance):
        return (float(p_ls) + pa) / 2.0, "consensus(ls+anova)"
    if agree(p_pdm, pa, tolerance):
        return (float(p_pdm) + pa) / 2.0, "consensus(pdm+anova)"
    if agree(p_bls, pa, tolerance):
        return (float(p_bls) + pa) / 2.0, "consensus(bls+anova)"

    if is_short and agree(p_pdm, p_bls, short_tol_pdm_bls) and agree(pa, p_pdm, short_tol_pdm_bls):
        return (float(p_pdm) + float(p_bls) + pa) / 3.0, "consensus(pdm+bls+anova)"

    return p_base, m_base


def _save_matplotlib_lc(lc: Any, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend (thread-safe-ish)
    with _MPL_LOCK:
        fig, ax = plt.subplots(figsize=(12, 4))
        lc.plot(ax=ax)
        ax.set_title(title)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)


def _save_matplotlib_phased(lc: Any, period_days: float, path: Path, title: str) -> None:
    if not (np.isfinite(period_days) and period_days > 0):
        return
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend (thread-safe-ish)
    with _MPL_LOCK:
        fig, ax = plt.subplots(figsize=(8, 6))
        folded = lc.fold(period_days * u.day, normalize_phase=True)
        folded.scatter(ax=ax)
        ax.set_title(title)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)


def _tpf_coord_deg(val: Any) -> float:
    try:
        if hasattr(val, "deg"):
            return float(val.deg)
        return float(val)
    except Exception:  # noqa: BLE001
        return float("nan")


def _extract_masterstar_cutout(
    fits_path: Path,
    ra0: float,
    dec0: float,
    size_arcmin: float = 5.0,
) -> tuple[np.ndarray, float, float, float, float] | None:
    """
    Vyreže cutout zo MASTERSTAR FITS okolo (ra0, dec0).
    Vracia (data, left_arcsec, right_arcsec, bottom_arcsec, top_arcsec)
    kde hodnoty sú ΔRA/ΔDec offsety rohov v arcsekundách od (ra0, dec0).
    Vracia None ak zlyhá.
    """
    try:
        from astropy.io import fits as afits
        from astropy.wcs import WCS
        from astropy.nddata import Cutout2D
        from astropy.coordinates import SkyCoord
        from astropy.wcs.utils import proj_plane_pixel_scales

        with afits.open(str(fits_path)) as hdul:
            hdu = next(
                (h for h in hdul if h.data is not None and h.data.ndim >= 2),
                None,
            )
            if hdu is None:
                logger.warning("[TESS blend] MASTERSTAR FITS: žiadne obrazové HDU")
                return None
            data = hdu.data.astype(float)
            if data.ndim > 2:
                data = data[0]
            wcs = WCS(hdu.header, naxis=2)

        # Pixel scale z WCS
        scales = proj_plane_pixel_scales(wcs)
        ps_arcsec = float(np.mean(scales) * 3600.0)
        if ps_arcsec <= 0 or ps_arcsec > 10.0:
            logger.warning("[TESS blend] MASTERSTAR pixel scale neplatná: %.2f\"/px", ps_arcsec)
            return None

        # Cutout
        coord = SkyCoord(ra=ra0, dec=dec0, unit="deg")
        size_px = int((size_arcmin * 60.0) / ps_arcsec)
        size_px = max(200, size_px)
        cutout = Cutout2D(data, coord, size_px, wcs=wcs, mode="partial", fill_value=np.nan)

        # Extent v arcsekundách od (0,0) = target
        h, w = cutout.data.shape
        half_w = (w / 2.0) * ps_arcsec
        half_h = (h / 2.0) * ps_arcsec

        logger.info(
            "[TESS blend] MASTERSTAR cutout: %dx%d px, %.2f\"/px, extent ±%.0f\" ±%.0f\"",
            w,
            h,
            ps_arcsec,
            half_w,
            half_h,
        )
        return cutout.data, -half_w, half_w, -half_h, half_h

    except Exception as exc:  # noqa: BLE001
        logger.warning("[TESS blend] MASTERSTAR cutout zlyhal: %s", exc)
        return None


def _generate_tess_blend_check_png(
    tpf: Any,
    catalog_id: str,
    target_mask: np.ndarray,
    output_dir: Path,
    sector: int,
    plate_scale_arcsec_px: float | None = None,
    gaia_db_path: Path | None = None,
    masterstar_fits_path: Path | None = None,
) -> Path | None:
    """
    Generuje blend-check PNG: TESS TPF (vľavo) + Gaia obloha (vpravo).

    Ľavý panel : TPF flux mapa + apertúra (červený obdĺžnik) + target (žltý krúžok)
    Pravý panel : Gaia hviezdy v okolí (bodky podľa mag) + TESS apertúra ako overlay
                  + pixel grid TESS + merítko 1'

    plate_scale_arcsec_px: z cfg.phase01_plate_scale_arcsec_per_px;
                           None = neznáma (zobrazí sa bez tejto informácie)
    gaia_db_path          : lokálna SQLite pre susedné hviezdy; None = len target
    masterstar_fits_path  : voliteľný MASTERSTAR FITS pre pozadie pravého panelu
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        TESS_ARCSEC_PX = 21.0  # TESS pixel scale arcsec/px

        flx = tpf.flux
        flux_cube = np.asarray(flx.value if hasattr(flx, "value") else flx, dtype=float)
        flux_med = np.nanmedian(flux_cube, axis=0)
        n_rows, n_cols = flux_med.shape
        if np.any(np.isfinite(flux_med)):
            bright_idx = np.unravel_index(int(np.nanargmax(flux_med)), flux_med.shape)
            center_row = int(bright_idx[0])
            center_col = int(bright_idx[1])
        else:
            center_row = n_rows // 2
            center_col = n_cols // 2
        apt_mask = np.asarray(target_mask, dtype=bool)

        fov_deg = (n_cols * TESS_ARCSEC_PX) / 3600.0 * 2.5
        # ra0/dec0 = stred TPF (pre Gaia query box)
        ra0 = _tpf_coord_deg(getattr(tpf, "ra", float("nan")))
        dec0 = _tpf_coord_deg(getattr(tpf, "dec", float("nan")))

        # ra_target/dec_target = presná poloha targetu (brightest pixel) cez TPF WCS
        try:
            _target_coord = tpf.wcs.pixel_to_world(center_col, center_row)
            ra_target = float(_target_coord.ra.deg)
            dec_target = float(_target_coord.dec.deg)
            logger.info("[TESS blend] target WCS: ra=%.6f, dec=%.6f", ra_target, dec_target)
        except Exception as _exc:  # noqa: BLE001
            logger.warning("[TESS blend] pixel_to_world zlyhal: %s — fallback na ra0/dec0", _exc)
            ra_target = ra0
            dec_target = dec0

        gaia_stars: list[dict[str, Any]] = []
        if gaia_db_path is None or not gaia_db_path.is_file():
            logger.warning(
                "[TESS blend] gaia_db_path nie je nastavená alebo súbor neexistuje: %s",
                gaia_db_path,
            )
        else:
            try:
                import sqlite3

                if np.isfinite(ra0) and np.isfinite(dec0):
                    con = sqlite3.connect(str(gaia_db_path))
                    try:
                        cur = con.execute(
                            """
                        SELECT source_id, ra, dec, g_mag FROM gaia_dr3
                        WHERE ra  BETWEEN ? AND ?
                          AND dec BETWEEN ? AND ?
                          AND g_mag <= 18.0
                        LIMIT 300
                        """,
                            (
                                ra0 - fov_deg,
                                ra0 + fov_deg,
                                dec0 - fov_deg,
                                dec0 + fov_deg,
                            ),
                        )
                        for row in cur.fetchall():
                            gaia_stars.append(
                                {
                                    "source_id": str(row[0]),
                                    "ra": float(row[1]),
                                    "dec": float(row[2]),
                                    "g_mag": float(row[3]) if row[3] is not None else 15.0,
                                }
                            )
                    finally:
                        con.close()
                    logger.info(
                        "[TESS blend] Gaia query: %d hviezd načítaných (ra0=%.4f, dec0=%.4f, fov_deg=%.4f)",
                        len(gaia_stars),
                        ra0,
                        dec0,
                        fov_deg,
                    )
                    if len(gaia_stars) == 0:
                        logger.warning(
                            "[TESS blend] Gaia query vrátila 0 hviezd (ra0=%.4f, dec0=%.4f, fov_deg=%.4f)",
                            ra0,
                            dec0,
                            fov_deg,
                        )
                else:
                    logger.warning(
                        "[TESS blend] Gaia query preskočená — neplatné ra0/dec0 z TPF (ra0=%s, dec0=%s)",
                        ra0,
                        dec0,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[TESS blend] Gaia query skipped: %s", exc)

        if plate_scale_arcsec_px is not None and float(plate_scale_arcsec_px) > 0:
            scope_label = f"Teleskop: {float(plate_scale_arcsec_px):.2f}\"/px"
        else:
            scope_label = "Teleskop: plate scale neznáma"

        with _MPL_LOCK:
            fig, (ax_tpf, ax_sky) = plt.subplots(1, 2, figsize=(22, 10), facecolor="#0e0e0e")
            fig.suptitle(
                f"{catalog_id}  |  Sektor {sector}  |  Blend check\n"
                f"TESS: {TESS_ARCSEC_PX:.0f}\"/px  ·  {scope_label}",
                color="white",
                fontsize=11,
            )

            ax_tpf.set_facecolor("#0e0e0e")
            im = ax_tpf.imshow(flux_med, origin="lower", cmap="YlOrRd", aspect="equal")
            plt.colorbar(im, ax=ax_tpf, label="Flux [e⁻/s]", fraction=0.046, pad=0.04)

            rows_apt, cols_apt = np.where(apt_mask)
            if len(rows_apt) > 0:
                r_min, r_max = rows_apt.min(), rows_apt.max()
                c_min, c_max = cols_apt.min(), cols_apt.max()
                ax_tpf.add_patch(
                    patches.Rectangle(
                        (c_min - 0.5, r_min - 0.5),
                        c_max - c_min + 1,
                        r_max - r_min + 1,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                        label="TESS apertúra",
                    )
                )

            ax_tpf.plot(
                center_col,
                center_row,
                "o",
                color="yellow",
                markersize=10,
                markeredgewidth=2,
                markerfacecolor="none",
                label="Target",
            )
            ax_tpf.set_title("TESS TPF", color="white", fontsize=10)
            ax_tpf.set_xlabel("Pixel (stĺpec)", color="white")
            ax_tpf.set_ylabel("Pixel (riadok)", color="white")
            ax_tpf.tick_params(colors="white")
            for sp in ax_tpf.spines.values():
                sp.set_edgecolor("gray")
            ax_tpf.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")

            ax_sky.set_facecolor("#0e0e0e")
            _ms_shown = False
            if masterstar_fits_path is not None and masterstar_fits_path.is_file() and np.isfinite(ra_target) and np.isfinite(dec_target):
                _ms_result = _extract_masterstar_cutout(masterstar_fits_path, ra_target, dec_target, size_arcmin=5.0)
                if _ms_result is not None:
                    _ms_data, _ext_left, _ext_right, _ext_bottom, _ext_top = _ms_result
                    from astropy.visualization import ZScaleInterval

                    _vmin, _vmax = ZScaleInterval().get_limits(_ms_data[np.isfinite(_ms_data)])
                    ax_sky.imshow(
                        _ms_data,
                        origin="lower",
                        cmap="gray",
                        vmin=_vmin,
                        vmax=_vmax,
                        extent=[_ext_left, _ext_right, _ext_bottom, _ext_top],
                        aspect="auto",
                        zorder=1,
                    )
                    _ms_shown = True
            if not _ms_shown:
                logger.info("[TESS blend] MASTERSTAR pozadie nedostupné — čierne pozadie")
            ax_sky.set_aspect("equal")
            half_fov = (n_cols * TESS_ARCSEC_PX) * 1.2
            ax_sky.set_xlim(-half_fov, half_fov)
            ax_sky.set_ylim(-half_fov, half_fov)
            ax_sky.invert_xaxis()

            if len(rows_apt) > 0:
                apt_w = (c_max - c_min + 1) * TESS_ARCSEC_PX
                apt_h = (r_max - r_min + 1) * TESS_ARCSEC_PX
                apt_center_col = (c_min + c_max) / 2.0
                apt_center_row = (r_min + r_max) / 2.0
                # Offset apertúry od target pixelu (nie od stredu TPF)
                # Po invert_xaxis: RA rastie doprava → znamienko kladné
                apt_x = (apt_center_col - center_col) * TESS_ARCSEC_PX - apt_w / 2.0
                apt_y = (apt_center_row - center_row) * TESS_ARCSEC_PX - apt_h / 2.0
                ax_sky.add_patch(
                    patches.Rectangle(
                        (apt_x, apt_y),
                        apt_w,
                        apt_h,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="red",
                        alpha=float(np.clip(0.15, 0.0, 1.0)),
                    )
                )
                ax_sky.add_patch(
                    patches.Rectangle(
                        (apt_x, apt_y),
                        apt_w,
                        apt_h,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                        label="TESS apertúra",
                    )
                )

            cos_dec = np.cos(np.radians(dec_target)) if np.isfinite(dec_target) else 1.0
            for star in gaia_stars:
                d_ra = (star["ra"] - ra_target) * 3600.0 * cos_dec
                d_dec = (star["dec"] - dec_target) * 3600.0
                g_mag = star["g_mag"]
                size = max(20, 200 - g_mag * 12)
                is_target = abs(d_ra) < 2 and abs(d_dec) < 2
                color = "yellow" if is_target else "lightcyan"
                alpha = float(np.clip(max(0.6, 1.0 - (g_mag - 10) / 8), 0.0, 1.0))
                ax_sky.scatter(d_ra, d_dec, s=size, c=color, alpha=alpha, zorder=5)

            ax_sky.plot(
                0,
                0,
                "o",
                color="yellow",
                markersize=14,
                markeredgewidth=2,
                markerfacecolor="none",
                label="Target",
                zorder=10,
            )

            _grid_alpha = float(np.clip(0.2, 0.0, 1.0))
            for i in np.arange(-half_fov, half_fov + 1e-6, TESS_ARCSEC_PX):
                ax_sky.axvline(float(i), color="gray", alpha=_grid_alpha, linewidth=0.5)
                ax_sky.axhline(float(i), color="gray", alpha=_grid_alpha, linewidth=0.5)

            scale_arcsec = 60.0
            sx = half_fov * 0.55
            sy = -half_fov * 0.85
            ax_sky.annotate(
                "",
                xy=(sx, sy),
                xytext=(sx - scale_arcsec, sy),
                arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.5),
            )
            ax_sky.text(
                sx - scale_arcsec / 2,
                sy + half_fov * 0.06,
                "1'",
                color="cyan",
                ha="center",
                fontsize=8,
            )

            ax_sky.set_title("Gaia obloha + TESS apertúra", color="white", fontsize=10)
            ax_sky.set_xlabel("ΔRA [arcsec]", color="white")
            ax_sky.set_ylabel("ΔDec [arcsec]", color="white")
            ax_sky.tick_params(colors="white")
            for sp in ax_sky.spines.values():
                sp.set_edgecolor("gray")
            ax_sky.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")

            plt.tight_layout()
            out_path = output_dir / f"sector_{sector}_blend_check.png"
            fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor="#0e0e0e")
            plt.close(fig)
        logger.info("[TESS blend] %s sektor %s → %s", catalog_id, sector, out_path.name)
        return out_path

    except Exception as exc:  # noqa: BLE001
        logger.warning("[TESS blend] failed for %s sektor %s: %s", catalog_id, sector, exc)
        return None


def _tpf_bjd_offset(tpf: Any) -> float:
    try:
        a = float(tpf.get_keyword("BJDREFI", hdu=1) or 0.0)
    except Exception:
        a = 0.0
    try:
        b = float(tpf.get_keyword("BJDREFF", hdu=1) or 0.0)
    except Exception:
        b = 0.0
    return a + b


def _process_one_sector(
    tpf: Any,
    catalog_id: str,
    mag: float | None,
    period_hint: float | None,
    output_dir: Path,
    cfg: Any | None = None,
) -> TessSectorResult:
    sector = int(getattr(tpf, "sector", 0) or 0)
    err_out = TessSectorResult(
        sector=sector,
        jd_start=float("nan"),
        jd_end=float("nan"),
        period_found=None,
        period_anova=None,
        period_2p=None,
        harmonic_note=None,
        n_points_before_sigma_clip=0,
        lc_raw_path=None,
        plot_raw_path=None,
        plot_phased_p_path=None,
        plot_phased_2p_path=None,
        n_points=0,
        error=None,
    )
    try:
        x, y, mx, my, cutsize = _get_aperture_params(mag)
        target_mask = tpf.create_threshold_mask(threshold=150, reference_pixel="center")
        bg_mask = ~tpf.create_threshold_mask(threshold=0.001, reference_pixel=None)
        target_mask = np.asarray(target_mask, dtype=bool).copy()
        target_mask[0:cutsize, 0:cutsize] = False
        target_mask[int(x) : int(x + mx), int(y) : int(y + my)] = True

        lc_t = tpf.to_lightcurve(aperture_mask=target_mask)
        lc_b = tpf.to_lightcurve(aperture_mask=bg_mask)
        n_bg = float(np.sum(bg_mask))
        n_t = float(np.sum(target_mask))
        if n_bg <= 0 or n_t <= 0:
            err_out.error = "Neplatná aperture maska (n_bg alebo n_t)."
            return err_out
        corr_lc = lc_t - ((lc_b / n_bg) * n_t)
        corr_lc = corr_lc.remove_nans()
        corr_lc = _delete_error(corr_lc, start=40, end=40, center=160)

        bjd_off = _tpf_bjd_offset(tpf)
        corr_lc.time = corr_lc.time + bjd_off

        # --- Pass 1: predbežná perióda na normalizovaných dátach bez flatten (ultrakrátké P) ---
        try:
            lc_norm = corr_lc.normalize()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[TESS] normalize (pass1) failed: %s", exc)
            lc_norm = corr_lc
        p_pre = _lomb_scargle_best_period(lc_norm, window_length=51, use_flatten=False)

        if period_hint is not None and np.isfinite(float(period_hint)) and float(period_hint) > 0:
            p_win: float | None = float(period_hint)
        elif p_pre is not None and np.isfinite(float(p_pre)) and float(p_pre) > 0:
            p_win = float(p_pre)
        else:
            p_win = None

        wl = _get_optimal_window(p_win)
        if int(wl) < 0:
            # Mira / P > 15 d: žiadny flatten — zachovať pomalý trend
            try:
                lc_search = corr_lc.normalize()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[TESS] normalize-only (long P) failed: %s", exc)
                lc_search = corr_lc
            wl_helpers = 101
        else:
            try:
                lc_search = corr_lc.flatten(window_length=int(wl))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[TESS] flatten pass2 failed (%s), fallback normalize", exc)
                try:
                    lc_search = corr_lc.normalize()
                except Exception:
                    lc_search = corr_lc
            wl_helpers = int(wl)

        err_out.n_points_before_sigma_clip = int(len(lc_search))
        lc_search = _iterative_sigma_clip_lc(lc_search)

        flux_arr = np.asarray(
            lc_search.flux.value if hasattr(lc_search.flux, "value") else lc_search.flux,
            dtype=float,
        )
        flux_std = float(np.std(flux_arr))
        amplitude_ppt = float((np.max(flux_arr) - np.min(flux_arr)) * 1000)
        snr = float(amplitude_ppt / (flux_std * 1000)) if flux_std > 0 else None
        err_out.amplitude_ppt = amplitude_ppt
        err_out.snr = snr
        err_out.flux_std = flux_std

        logger.info(
            "[TESS] Sector %d two-pass: P_pre=%s P_win=%s flatten_wl=%s (-1=normalize only)",
            int(sector),
            f"{p_pre:.6g}" if p_pre is not None and np.isfinite(float(p_pre)) else str(p_pre),
            f"{p_win:.6g}" if p_win is not None and np.isfinite(float(p_win)) else str(p_win),
            str(wl),
        )

        # Pass 2: LS / PDM / BLS / ANOVA(astropy LS) na lc_search (normalize + dynamický flatten v _prepare_lc_for_period)
        try:
            with ThreadPoolExecutor(max_workers=4) as ex:
                f_ls = ex.submit(
                    _find_period,
                    lc_search,
                    window_length=wl_helpers,
                    period_hint=period_hint,
                    ignore_period_hint=False,
                    use_flatten_for_ls=True,
                )
                f_pdm = ex.submit(
                    _find_period_pdm,
                    lc_search,
                    wl_helpers,
                    use_flatten=True,
                )
                f_bls = ex.submit(
                    _find_period_bls,
                    lc_search,
                    wl_helpers,
                    use_flatten=True,
                )
                f_anova = ex.submit(
                    _find_period_anova,
                    lc_search,
                    wl_helpers,
                    use_flatten=True,
                )
                period_ls = f_ls.result()
                period_pdm = f_pdm.result()
                period_bls = f_bls.result()
                period_anova = f_anova.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[TESS] period threadpool failed, fallback to sequential: %s", exc)
            period_ls = _find_period(
                lc_search,
                window_length=wl_helpers,
                period_hint=period_hint,
                use_flatten_for_ls=True,
            )
            period_pdm = _find_period_pdm(lc_search, wl_helpers, use_flatten=True)
            period_bls = _find_period_bls(lc_search, wl_helpers, use_flatten=True)
            period_anova = _find_period_anova(lc_search, wl_helpers, use_flatten=True)
        period_consensus, method_used = _period_consensus(
            period_ls,
            period_pdm,
            period_bls,
            period_anova,
        )
        period_consensus, harmonic_tag = _harmonic_refine_period(lc_search, period_consensus)
        if harmonic_tag:
            method_used = f"{method_used}|{harmonic_tag}"
            err_out.harmonic_note = str(harmonic_tag)
        else:
            err_out.harmonic_note = None

        logger.info(
            "[TESS] Sector %d periods: LS=%.4f PDM=%.4f BLS=%.4f ANOVA=%.4f → consensus=%.4f (%s)",
            int(sector),
            float(period_ls) if period_ls is not None and np.isfinite(period_ls) else -1.0,
            float(period_pdm) if period_pdm is not None and np.isfinite(period_pdm) else -1.0,
            float(period_bls) if period_bls is not None and np.isfinite(period_bls) else -1.0,
            float(period_anova) if period_anova is not None and np.isfinite(period_anova) else -1.0,
            float(period_consensus) if period_consensus is not None and np.isfinite(period_consensus) else -1.0,
            str(method_used),
        )

        err_out.period_found = period_ls
        err_out.period_pdm = period_pdm
        err_out.period_bls = period_bls
        err_out.period_anova = period_anova
        err_out.period_consensus = period_consensus
        err_out.period_method_used = str(method_used)
        err_out.period_2p = (
            (float(period_consensus) * 2.0)
            if period_consensus is not None and np.isfinite(period_consensus) and float(period_consensus) > 0
            else None
        )

        # Výstupná krivka = rovnaká ako pri pass-2 vyhľadávaní (adaptívny flatten alebo len normalize)
        lc_out = lc_search

        t_arr = np.asarray(lc_out.time.value if hasattr(lc_out.time, "value") else lc_out.time, dtype=float)
        err_out.jd_start = float(np.nanmin(t_arr)) if t_arr.size else float("nan")
        err_out.jd_end = float(np.nanmax(t_arr)) if t_arr.size else float("nan")
        err_out.n_points = int(len(lc_out))

        raw_path = output_dir / f"sector_{sector}_raw.csv"
        try:
            lc_out.to_csv(str(raw_path), overwrite=True)
        except TypeError:
            lc_out.to_csv(str(raw_path))
        err_out.lc_raw_path = str(raw_path)

        p_disp = float(period_consensus) if period_consensus is not None and np.isfinite(period_consensus) else float("nan")
        title_lc = f"{catalog_id} | Sektor {sector} | P={p_disp:.6f}d" if np.isfinite(p_disp) else f"{catalog_id} | Sektor {sector}"
        plot_lc = output_dir / f"sector_{sector}_lc.png"
        _save_matplotlib_lc(lc_out, plot_lc, title_lc)
        err_out.plot_raw_path = str(plot_lc)

        _gaia_db: Path | None = None
        if cfg is not None:
            _gp = str(getattr(cfg, "gaia_db_path", "") or "").strip()
            if _gp:
                _gaia_db = Path(_gp)
        _ms_fits: Path | None = None
        # Štruktúra: output_dir = .../NoFilter_60_2/photometry
        # MASTERSTAR.fits je v .../NoFilter_60_2/
        _ms_candidates = (
            list(output_dir.glob("MASTERSTAR*.fits"))
            + list(output_dir.parent.glob("MASTERSTAR*.fits"))
            + list(output_dir.parent.parent.glob("MASTERSTAR*.fits"))
            + list(output_dir.parent.parent.parent.glob("MASTERSTAR*.fits"))
        )
        if _ms_candidates:
            _ms_fits = _ms_candidates[0]
            logger.info("[TESS blend] MASTERSTAR FITS: %s", _ms_fits)
        else:
            logger.warning("[TESS blend] MASTERSTAR FITS nenájdený v okolí %s", output_dir)

        _plate_scale_ref: float | None = None
        # Authoritative: solved WCS/CD from MASTERSTAR via the shared resolver
        # (config phase01_plate_scale_arcsec_per_px is only a last resort there).
        if _ms_fits is not None:
            try:
                from photometry_core import _read_plate_scale_from_fits_path  # noqa: PLC0415

                _ps_res = _read_plate_scale_from_fits_path(Path(_ms_fits))
                if _ps_res is not None and float(_ps_res) > 0:
                    _plate_scale_ref = float(_ps_res)
            except Exception:  # noqa: BLE001
                _plate_scale_ref = None
        if _plate_scale_ref is None and cfg is not None:
            _psv = float(getattr(cfg, "phase01_plate_scale_arcsec_per_px", 0) or 0)
            _plate_scale_ref = _psv if _psv > 0 else None
        # plate_scale pre label v nadpise — ak None, zobrazí sa "neznáma"
        # extent výpočet používa WCS priamo v _extract_masterstar_cutout

        blend_path = _generate_tess_blend_check_png(
            tpf=tpf,
            catalog_id=str(catalog_id),
            target_mask=target_mask,
            output_dir=output_dir,
            sector=sector,
            plate_scale_arcsec_px=_plate_scale_ref,
            gaia_db_path=_gaia_db,
            masterstar_fits_path=_ms_fits,
        )
        if blend_path is not None:
            err_out.blend_check_path = str(blend_path)

        if period_consensus is not None and np.isfinite(period_consensus) and float(period_consensus) > 0:
            p = float(period_consensus)
            pp = output_dir / f"sector_{sector}_phased_P.png"
            _save_matplotlib_phased(
                lc_out,
                p,
                pp,
                f"{catalog_id} | Sektor {sector} | P={p:.6f}d",
            )
            err_out.plot_phased_p_path = str(pp)
            p2p = output_dir / f"sector_{sector}_phased_2P.png"
            _save_matplotlib_phased(
                lc_out,
                p * 2.0,
                p2p,
                f"{catalog_id} | Sektor {sector} | 2P={p * 2.0:.6f}d",
            )
            err_out.plot_phased_2p_path = str(p2p)
    except Exception as exc:  # noqa: BLE001
        err_out.error = str(exc)
        logger.exception("TESS sector %s failed", sector)
    return err_out


def _assess_period_reliability(
    consensus_periods: list[float],
    sectors: list[TessSectorResult],
    snr_threshold: float = 5.0,
    agree_tolerance: float = 0.20,
) -> tuple[float | None, str, str]:
    """
    Vypočíta robustný globálny period_consensus a reliability flag.

    Vracia: (p_con, reliability, reason)

    Pravidlá:
    - noise:     žiadne consensus_periods, alebo max SNR < snr_threshold
    - uncertain: len 1 sektor, alebo sektory sa nezhodujú (>20% rozdiel)
    - reliable:  ≥2 sektory sa zhodujú v tolerancii 20%
    """
    if not consensus_periods:
        return None, "noise", "Žiadny sektor nedal konsenzus periódy"

    snr_vals = [
        float(s.snr) for s in sectors if s.snr is not None and np.isfinite(float(s.snr))
    ]
    max_snr = max(snr_vals) if snr_vals else 0.0
    if max_snr < snr_threshold:
        return None, "noise", f"Max SNR={max_snr:.1f} < prah {snr_threshold:.1f}"

    if len(consensus_periods) == 1:
        return float(consensus_periods[0]), "uncertain", "Len 1 sektor s konsenzom"

    p_sorted = sorted(consensus_periods)
    agree_pairs: list[tuple[float, float]] = []
    for i in range(len(p_sorted)):
        for j in range(i + 1, len(p_sorted)):
            p1, p2 = p_sorted[i], p_sorted[j]
            rel_diff = abs(p1 - p2) / ((p1 + p2) / 2)
            if rel_diff <= agree_tolerance:
                agree_pairs.append((p1, p2))

    if agree_pairs:
        agreed_vals = list({v for pair in agree_pairs for v in pair})
        p_con = float(statistics.median(agreed_vals))
        return (
            p_con,
            "reliable",
            f"Sektory v zhode (tol={agree_tolerance * 100:.0f}%): "
            f"{[f'{v:.4f}' for v in agreed_vals]}",
        )

    p_con = float(statistics.median(consensus_periods))
    return (
        p_con,
        "uncertain",
        f"Sektory sa nezhodujú: {[f'{v:.4f}' for v in p_sorted]} "
        f"(max rozdiel > {agree_tolerance * 100:.0f}%)",
    )


def run_tess_analysis(
    catalog_id: str,
    ra: float,
    dec: float,
    mag: float | None,
    photometry_dir: str,
    period_hint: float | None,
    progress_callback: Callable[[str, float], None] | None = None,
    cfg: Any | None = None,
) -> TessResult:
    from config import AppConfig  # noqa: PLC0415

    _cfg = cfg if cfg is not None else AppConfig()
    cid = str(catalog_id).strip()
    out_base = Path(photometry_dir).resolve() / "_tess" / cid
    out_base.mkdir(parents=True, exist_ok=True)

    if not bool(getattr(_cfg, "tess_enabled", False)):
        logging.info("[TESS] preskočené — tess_enabled=False (config.AppConfig / config.json)")
        return TessResult(
            catalog_id=cid,
            ra=float(ra),
            dec=float(dec),
            mag=mag,
            output_dir=str(out_base),
            total_sectors_found=0,
            total_sectors_ok=0,
            error_global="TESS je v konfigurácii vypnutý (tess_enabled=False). Nastav v config.json: \"tess_enabled\": true",
        )

    import lightkurve as lk

    def _prog(msg: str, val: float) -> None:
        if progress_callback is not None:
            try:
                progress_callback(msg, float(val))
            except Exception:  # noqa: BLE001
                pass

    x, y, mx, my, cutsize = _get_aperture_params(mag)

    try:
        search_results = lk.search_tesscut(f"{float(ra)} {float(dec)}", sector=list(range(1, 70)))
    except Exception as exc:  # noqa: BLE001
        return TessResult(
            catalog_id=cid,
            ra=float(ra),
            dec=float(dec),
            mag=mag,
            output_dir=str(out_base),
            error_global=f"Chyba vyhľadania TESS: {exc}",
        )

    n_found = int(len(search_results))
    if n_found == 0:
        return TessResult(
            catalog_id=cid,
            ra=float(ra),
            dec=float(dec),
            mag=mag,
            output_dir=str(out_base),
            total_sectors_found=0,
            total_sectors_ok=0,
            error_global="Žiadne TESS dáta pre túto pozíciu",
        )

    _prog(f"Nájdených {n_found} sektorov, sťahujem...", 0.1)

    try:
        tpfs = search_results.download_all(cutout_size=int(cutsize), quality_bitmask="hard")
    except Exception as exc:  # noqa: BLE001
        return TessResult(
            catalog_id=cid,
            ra=float(ra),
            dec=float(dec),
            mag=mag,
            output_dir=str(out_base),
            total_sectors_found=n_found,
            error_global=f"Chyba sťahovania TESS: {exc}",
        )

    _prog("Stiahnuté, spracovávam sektory...", 0.3)

    # Parallelize per-sector processing. Sectors are independent.
    MAX_SECTOR_WORKERS = 4
    sectors: list[TessSectorResult] = []
    n_tpfs = max(1, int(len(tpfs)))

    with ThreadPoolExecutor(max_workers=int(MAX_SECTOR_WORKERS)) as executor:
        future_to_tpf = {
            executor.submit(
                _process_one_sector,
                tpf,
                cid,
                mag,
                period_hint,
                out_base,
                _cfg,
            ): tpf
            for tpf in tpfs
        }

        n_done = 0
        for future in as_completed(future_to_tpf):
            n_done += 1
            frac = 0.3 + 0.6 * (float(n_done) / float(n_tpfs))
            _prog(f"Sektor {n_done}/{n_tpfs} hotový", min(0.99, frac))
            try:
                result = future.result()
                sectors.append(result)
            except Exception as exc:  # noqa: BLE001
                tpf0 = future_to_tpf.get(future)
                sector_num = getattr(tpf0, "sector", "?") if tpf0 is not None else "?"
                logging.warning("[TESS] Sector %s failed: %s", sector_num, exc)
                try:
                    sec_i = int(sector_num) if str(sector_num).isdigit() else 0
                except Exception:
                    sec_i = 0
                sectors.append(TessSectorResult(sector=sec_i, jd_start=0.0, jd_end=0.0, error=str(exc)))

    sectors.sort(key=lambda s: int(s.sector))

    consensus_periods: list[float] = []
    anova_periods: list[float] = []
    n_ok = 0
    for s in sectors:
        if (
            s.error is None
            and s.period_consensus is not None
            and np.isfinite(s.period_consensus)
            and float(s.period_consensus) > 0
        ):
            consensus_periods.append(float(s.period_consensus))
        if (
            s.error is None
            and s.period_anova is not None
            and np.isfinite(s.period_anova)
            and float(s.period_anova) > 0
        ):
            anova_periods.append(float(s.period_anova))
        if s.error is None:
            n_ok += 1

    p_con, p_reliability, p_reliability_reason = _assess_period_reliability(consensus_periods, sectors)

    p_anova_con: float | None = None
    if anova_periods:
        p_anova_con = float(statistics.median(anova_periods))

    p2_con = (p_con * 2.0) if p_con is not None and np.isfinite(p_con) else None

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    summary = {
        "catalog_id": cid,
        "ra": float(ra),
        "dec": float(dec),
        "mag": mag,
        "period_consensus": p_con,
        "period_anova_consensus": p_anova_con,
        "period_2p_consensus": p2_con,
        "period_reliability": p_reliability,
        "period_reliability_reason": p_reliability_reason,
        "total_sectors_found": n_found,
        "total_sectors_ok": int(n_ok),
        "sectors": [],
    }
    for s in sectors:
        summary["sectors"].append(
            {
                "sector": s.sector,
                "jd_start": s.jd_start,
                "jd_end": s.jd_end,
                "period_ls": s.period_found,
                "period_pdm": s.period_pdm,
                "period_bls": s.period_bls,
                "period_anova": s.period_anova,
                "period_consensus": s.period_consensus,
                "period_method_used": s.period_method_used,
                "harmonic_note": s.harmonic_note,
                "period_2p": s.period_2p,
                "n_points": s.n_points,
                "n_points_before_sigma_clip": s.n_points_before_sigma_clip,
                "lc_raw_path": s.lc_raw_path,
                "plot_raw_path": s.plot_raw_path,
                "plot_phased_p_path": s.plot_phased_p_path,
                "plot_phased_2p_path": s.plot_phased_2p_path,
                "blend_check_path": s.blend_check_path,
                "error": s.error,
                "amplitude_ppt": s.amplitude_ppt,
                "snr": s.snr,
                "flux_std": s.flux_std,
            }
        )
    with (out_base / "result.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)

    return TessResult(
        catalog_id=cid,
        ra=float(ra),
        dec=float(dec),
        mag=mag,
        sectors=sectors,
        period_consensus=p_con,
        period_anova_consensus=p_anova_con,
        period_2p_consensus=p2_con,
        output_dir=str(out_base),
        total_sectors_found=n_found,
        total_sectors_ok=int(n_ok),
        error_global=None,
        period_reliability=p_reliability,
        period_reliability_reason=p_reliability_reason,
    )
