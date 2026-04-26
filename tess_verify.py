"""
Download and analyze TESS cutout light curves for variable-star candidates (lightkurve).
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

logger = logging.getLogger(__name__)


@dataclass
class TessSectorResult:
    sector: int
    jd_start: float
    jd_end: float
    period_found: float | None
    period_2p: float | None
    lc_raw_path: str | None
    plot_raw_path: str | None
    plot_phased_p_path: str | None
    plot_phased_2p_path: str | None
    n_points: int
    error: str | None = None


@dataclass
class TessResult:
    catalog_id: str
    ra: float
    dec: float
    mag: float | None
    sectors: list[TessSectorResult] = field(default_factory=list)
    period_consensus: float | None = None
    period_2p_consensus: float | None = None
    output_dir: str = ""
    total_sectors_found: int = 0
    total_sectors_ok: int = 0
    error_global: str | None = None

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
        p2 = self.period_2p_consensus
        ps = f"P={p:.6g} d" if p is not None else "P=—"
        p2s = f"2P={p2:.6g} d" if p2 is not None else "2P=—"
        return (
            f"TESS: {self.total_sectors_ok}/{self.total_sectors_found} sektorov OK | {ps} | {p2s} | {self.output_dir}"
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


def _find_period(lc: Any, window_length: int = 101, period_hint: float | None = None) -> float | None:
    if period_hint is not None and np.isfinite(period_hint) and float(period_hint) > 0:
        return float(period_hint)
    list_section = [0.05, 0.1, 0.25, 0.5, 1.1, 2, 5, 10, 50]
    try:
        flat = lc.flatten(window_length=int(window_length))
    except Exception as exc:  # noqa: BLE001
        logger.warning("flatten failed in _find_period: %s", exc)
        return None
    best_pg = None
    best_power = -1.0
    for i in range(len(list_section) - 1):
        p0, p1 = float(list_section[i]), float(list_section[i + 1])
        if not (np.isfinite(p0) and np.isfinite(p1) and p1 > p0):
            continue
        try:
            pg = flat.to_periodogram(
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


def _save_matplotlib_lc(lc: Any, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    lc.plot(ax=ax)
    ax.set_title(title)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_matplotlib_phased(lc: Any, period_days: float, path: Path, title: str) -> None:
    if not (np.isfinite(period_days) and period_days > 0):
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    folded = lc.fold(period_days * u.day, normalize_phase=True)
    folded.scatter(ax=ax)
    ax.set_title(title)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    *,
    catalog_id: str,
    output_dir: Path,
    cutsize: int,
    x: int,
    y: int,
    mx: int,
    my: int,
    period_hint: float | None,
) -> TessSectorResult:
    sector = int(getattr(tpf, "sector", 0) or 0)
    err_out = TessSectorResult(
        sector=sector,
        jd_start=float("nan"),
        jd_end=float("nan"),
        period_found=None,
        period_2p=None,
        lc_raw_path=None,
        plot_raw_path=None,
        plot_phased_p_path=None,
        plot_phased_2p_path=None,
        n_points=0,
        error=None,
    )
    try:
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

        period = _find_period(corr_lc, window_length=101, period_hint=period_hint)
        err_out.period_found = period
        err_out.period_2p = (period * 2.0) if period is not None and np.isfinite(period) and period > 0 else None

        wl = 101 if (period is not None and np.isfinite(period) and float(period) < 1.0) else 301
        try:
            lc_out = corr_lc.flatten(window_length=int(wl))
        except Exception:
            lc_out = corr_lc

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

        p_disp = float(period) if period is not None and np.isfinite(period) else float("nan")
        title_lc = f"{catalog_id} | Sektor {sector} | P={p_disp:.6f}d" if np.isfinite(p_disp) else f"{catalog_id} | Sektor {sector}"
        plot_lc = output_dir / f"sector_{sector}_lc.png"
        _save_matplotlib_lc(lc_out, plot_lc, title_lc)
        err_out.plot_raw_path = str(plot_lc)

        if period is not None and np.isfinite(period) and float(period) > 0:
            p = float(period)
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


def run_tess_analysis(
    catalog_id: str,
    ra: float,
    dec: float,
    mag: float | None,
    photometry_dir: str,
    period_hint: float | None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> TessResult:
    import lightkurve as lk

    cid = str(catalog_id).strip()
    out_base = Path(photometry_dir).resolve() / "_tess" / cid
    out_base.mkdir(parents=True, exist_ok=True)

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

    sectors: list[TessSectorResult] = []
    n_tpfs = max(1, len(tpfs))
    for i, tpf in enumerate(tpfs, start=1):
        sec = _process_one_sector(
            tpf,
            catalog_id=cid,
            output_dir=out_base,
            cutsize=int(cutsize),
            x=x,
            y=y,
            mx=mx,
            my=my,
            period_hint=period_hint,
        )
        sectors.append(sec)
        frac = 0.3 + 0.6 * (float(i) / float(n_tpfs))
        _prog(f"Sektor {sec.sector} hotový", min(0.99, frac))

    ok_periods: list[float] = []
    n_ok = 0
    for s in sectors:
        if s.error is None and s.period_found is not None and np.isfinite(s.period_found) and s.period_found > 0:
            ok_periods.append(float(s.period_found))
        if s.error is None:
            n_ok += 1

    p_con: float | None = None
    if len(ok_periods) >= 2:
        p_con = float(statistics.median(ok_periods))
    elif len(ok_periods) == 1:
        p_con = float(ok_periods[0])

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
        "period_2p_consensus": p2_con,
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
                "period_found": s.period_found,
                "period_2p": s.period_2p,
                "n_points": s.n_points,
                "lc_raw_path": s.lc_raw_path,
                "plot_raw_path": s.plot_raw_path,
                "plot_phased_p_path": s.plot_phased_p_path,
                "plot_phased_2p_path": s.plot_phased_2p_path,
                "error": s.error,
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
        period_2p_consensus=p2_con,
        output_dir=str(out_base),
        total_sectors_found=n_found,
        total_sectors_ok=int(n_ok),
        error_global=None,
    )
