"""
Standalone cross-match of variable-star candidates against public catalogs (SIMBAD, Vizier).
"""
from __future__ import annotations

import math
import warnings
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter("ignore", AstropyWarning)
warnings.simplefilter("ignore", category=UserWarning, append=True)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s or s in ("--", "---", "nan", "NaN", "…", "..."):
            return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _sep_arcsec(coord: SkyCoord, ra_deg: Any, dec_deg: Any) -> float | None:
    try:
        c2 = SkyCoord(ra=_safe_float(ra_deg) * u.deg, dec=_safe_float(dec_deg) * u.deg, frame="icrs")
        if c2.ra.deg is None or c2.dec.deg is None:
            return None
        return float(coord.separation(c2).to(u.arcsec).value)
    except Exception:
        return None


@dataclass
class CatalogMatch:
    catalog: str
    name: str
    var_type: str = ""
    period: float | None = None
    amplitude: float | None = None
    delta_r: float | None = None
    mag: float | None = None
    epoch: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        parts: list[str] = []
        if self.var_type:
            parts.append(str(self.var_type))
        if self.period is not None:
            parts.append(f"P={self.period:.4g}d")
        if self.amplitude is not None:
            parts.append(f"amp={self.amplitude:.3g}")
        if self.delta_r is not None:
            parts.append(f"dr={self.delta_r:.2f}\"")
        return " · ".join(parts) if parts else (self.name or "—")


@dataclass
class CrossmatchResult:
    ra: float
    dec: float
    mag: float | None
    radius_arcsec: float
    matches: dict[str, list[CatalogMatch]]
    errors: dict[str, str]

    def has_any_match(self) -> bool:
        return any(bool(v) for v in self.matches.values())

    def best_period(self) -> float | None:
        order = ("VSX", "ASAS-SN", "ZTF", "Gaia varisum", "ATLAS", "CSS", "KELT", "VSBS", "TESS-EB", "SIMBAD")
        for cat in order:
            for m in self.matches.get(cat, []):
                if m.period is not None and math.isfinite(m.period) and m.period > 0:
                    return float(m.period)
        for cat, lst in self.matches.items():
            if cat in order:
                continue
            for m in lst:
                if m.period is not None and math.isfinite(m.period) and m.period > 0:
                    return float(m.period)
        return None

    def catalog_summary_bullets(self) -> list[str]:
        lines: list[str] = []
        preferred = [
            "SIMBAD",
            "VSX",
            "ASAS-SN",
            "ZTF",
            "Gaia varisum",
            "ATLAS",
            "CSS",
            "KELT",
            "VSBS",
            "TESS-EB",
        ]
        seen: set[str] = set()
        for cat in preferred:
            seen.add(cat)
            lst = self.matches.get(cat, [])
            err = self.errors.get(cat)
            if err:
                lines.append(f"{cat}: chyba ({err})")
            elif not lst:
                lines.append(f"{cat}: žiadny záznam")
            else:
                head = lst[0]
                tail = f" (+{len(lst) - 1} ďalších)" if len(lst) > 1 else ""
                lines.append(f"{cat}: {head.name or '—'} — {head.summary()}{tail}")
        for cat in sorted(self.matches.keys()):
            if cat in seen:
                continue
            lst = self.matches.get(cat, [])
            err = self.errors.get(cat)
            if err:
                lines.append(f"{cat}: chyba ({err})")
            elif not lst:
                lines.append(f"{cat}: žiadny záznam")
            else:
                head = lst[0]
                tail = f" (+{len(lst) - 1} ďalších)" if len(lst) > 1 else ""
                lines.append(f"{cat}: {head.name or '—'} — {head.summary()}{tail}")
        return lines


# --- Gaia varisum flag semantics (Gaia DR3 variability summary) ---
_GAIA_VARISUM_FLAGS: dict[str, str] = {
    "VRRLyr": "RR Lyrae-like",
    "VCep": "Cepheid-like",
    "VEB": "Eclipsing / ellipsoidal",
    "VLPV": "Long-period variable",
    "VST": "Short-timescale variable",
    "VRM": "Rotation modulation",
    "VMSO": "Main-sequence oscillator",
    "VAGN": "AGN / extragalactic candidate",
    "VCC": "Close companion / contamination",
    "VPN": "Stochastic / instrumental variability",
}


def _gaia_varisum_types(row: Any) -> str:
    active: list[str] = []
    for col, label in _GAIA_VARISUM_FLAGS.items():
        try:
            v = row[col]
        except Exception:
            continue
        ok = False
        if isinstance(v, (bool, np.bool_)):
            ok = bool(v)
        else:
            fv = _safe_float(v)
            if fv is not None and fv > 0:
                ok = True
            elif isinstance(v, str) and v.strip() not in ("", "0", "N", "n", "F", "f", "--"):
                ok = True
        if ok:
            active.append(label)
    return ", ".join(active) if active else ""


def _row_get(row: Any, *names: str) -> Any:
    cols = getattr(row, "colnames", None)
    for n in names:
        if cols is not None and n not in cols:
            continue
        try:
            return row[n]
        except Exception:
            continue
    return None


def _query_simbad(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    from astroquery.simbad import Simbad

    s = Simbad()
    try:
        s.add_votable_fields("otype", "V")
    except Exception:
        s.add_votable_fields("otype")
    res = s.query_region(coord, radius=radius_arcsec * u.arcsec)
    if res is None or len(res) == 0:
        return []
    out: list[CatalogMatch] = []
    for row in res:
        name = str(_row_get(row, "MAIN_ID", "main_id") or "").strip()
        otype = str(_row_get(row, "OTYPE", "otype") or "").strip()
        mag = _safe_float(_row_get(row, "V", "FLUX_V", "flux(V)"))
        dr = None
        try:
            cra = _row_get(row, "RA", "ra", "RA_ICRS")
            cde = _row_get(row, "DEC", "dec", "DE_ICRS")
            dr = _sep_arcsec(coord, cra, cde)
        except Exception:
            dr = None
        out.append(
            CatalogMatch(
                catalog="SIMBAD",
                name=name or "—",
                var_type=otype,
                mag=mag,
                delta_r=dr,
                extra={"raw_otype": otype},
            )
        )
    return out


def _vizier_radius_str(radius_arcsec: float) -> str:
    return f"{max(1, int(round(float(radius_arcsec))))}s"


def _vizier_table(coord: SkyCoord, radius_arcsec: float, catalog: str, columns: list[str] | None = None):
    from astroquery.vizier import Vizier

    v = Vizier(row_limit=10)
    if columns:
        v.columns = columns
    return v.query_region(coord, radius=_vizier_radius_str(radius_arcsec), catalog=catalog)


def _query_vsx(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    tl = _vizier_table(
        coord,
        radius_arcsec,
        "B/vsx/vsx",
        ["Name", "Type", "Period", "max", "min", "Epoch", "RAJ2000", "DEJ2000"],
    )
    if not tl or len(tl) == 0:
        return []
    t = tl[0]
    out: list[CatalogMatch] = []
    for row in t:
        name = str(_row_get(row, "Name") or "").strip()
        vtype = str(_row_get(row, "Type") or "").strip()
        per = _safe_float(_row_get(row, "Period"))
        mx = _safe_float(_row_get(row, "max"))
        mn = _safe_float(_row_get(row, "min"))
        amp = None
        if mx is not None and mn is not None:
            amp = abs(mx - mn)
        ep = _safe_float(_row_get(row, "Epoch"))
        dr = _sep_arcsec(coord, _row_get(row, "RAJ2000"), _row_get(row, "DEJ2000"))
        mag = mx if mx is not None else mn
        out.append(
            CatalogMatch(
                catalog="VSX",
                name=name,
                var_type=vtype,
                period=per,
                amplitude=amp,
                epoch=ep,
                mag=mag,
                delta_r=dr,
            )
        )
    return out


def _query_asassn(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    tl = _vizier_table(
        coord,
        radius_arcsec,
        "II/366",
        ["ASASSN-V", "Vmag", "Amp", "Per", "Type", "HJD", "RAJ2000", "DEJ2000"],
    )
    if not tl or len(tl) == 0:
        return []
    t = tl[0]
    out: list[CatalogMatch] = []
    for row in t:
        name = str(_row_get(row, "ASASSN-V") or "").strip()
        out.append(
            CatalogMatch(
                catalog="ASAS-SN",
                name=name,
                var_type=str(_row_get(row, "Type") or "").strip(),
                period=_safe_float(_row_get(row, "Per")),
                amplitude=_safe_float(_row_get(row, "Amp")),
                mag=_safe_float(_row_get(row, "Vmag")),
                epoch=_safe_float(_row_get(row, "HJD")),
                delta_r=_sep_arcsec(coord, _row_get(row, "RAJ2000"), _row_get(row, "DEJ2000")),
            )
        )
    return out


def _query_ztf(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    # Vizier meta uses table2 for periodic variables (paper ApJS 249/18).
    tl = _vizier_table(
        coord,
        radius_arcsec,
        "J/ApJS/249/18/table2",
        ["ID", "rmag", "Per", "Type", "rAmp", "gAmp", "RAJ2000", "DEJ2000"],
    )
    if not tl or len(tl) == 0:
        return []
    t = tl[0]
    out: list[CatalogMatch] = []
    for row in t:
        name = str(_row_get(row, "ID") or "").strip()
        ramp = _safe_float(_row_get(row, "rAmp"))
        gamp = _safe_float(_row_get(row, "gAmp"))
        amp = ramp if ramp is not None else gamp
        out.append(
            CatalogMatch(
                catalog="ZTF",
                name=name,
                var_type=str(_row_get(row, "Type") or "").strip(),
                period=_safe_float(_row_get(row, "Per")),
                amplitude=amp,
                mag=_safe_float(_row_get(row, "rmag")),
                delta_r=_sep_arcsec(coord, _row_get(row, "RAJ2000"), _row_get(row, "DEJ2000")),
            )
        )
    return out


def _query_gaia_varisum(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    tl = _vizier_table(coord, radius_arcsec, "I/358/varisum", None)
    if not tl or len(tl) == 0:
        return []
    t = tl[0]
    out: list[CatalogMatch] = []
    for row in t:
        sid = str(_row_get(row, "Source") or "").strip()
        gmag = _safe_float(_row_get(row, "Gmagmean"))
        flags = _gaia_varisum_types(row)
        out.append(
            CatalogMatch(
                catalog="Gaia varisum",
                name=sid,
                var_type=flags or "—",
                mag=gmag,
                delta_r=_sep_arcsec(coord, _row_get(row, "RA_ICRS"), _row_get(row, "DE_ICRS")),
                extra={k: _row_get(row, k) for k in _GAIA_VARISUM_FLAGS if k in getattr(row, "colnames", [])},
            )
        )
    return out


def _query_atlas(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    tl = _vizier_table(coord, radius_arcsec, "J/AJ/156/241/table4", None)
    if not tl or len(tl) == 0:
        return []
    t = tl[0]
    out: list[CatalogMatch] = []
    for row in t:
        oid = str(_row_get(row, "ATOID", "ID") or "").strip()
        per = _safe_float(_row_get(row, "fp-period"))
        mn = _safe_float(_row_get(row, "fp-min-o"))
        mx = _safe_float(_row_get(row, "fp-max-o"))
        amp = abs(mx - mn) if (mx is not None and mn is not None) else None
        mag = _safe_float(_row_get(row, "df-meanPvar"))
        if mag is None or mag <= 0.0 or mag > 30.0:
            mag = mn
        if mag is not None and (mag <= 0.0 or mag > 30.0):
            mag = None
        out.append(
            CatalogMatch(
                catalog="ATLAS",
                name=oid,
                var_type="",
                period=per,
                amplitude=amp,
                mag=mag,
                delta_r=_sep_arcsec(coord, _row_get(row, "RAJ2000"), _row_get(row, "DEJ2000")),
                extra={"columns": list(getattr(t, "colnames", []))},
            )
        )
    return out


def _query_css(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    # J/AJ/147/119 is Catalina Serpens periodic sample; schema uses KIC/KOI, not "CSS" id.
    tl = _vizier_table(
        coord,
        radius_arcsec,
        "J/AJ/147/119/table1",
        None,
    )
    if not tl or len(tl) == 0:
        return []
    t = tl[0]
    out: list[CatalogMatch] = []
    for row in t:
        kid = _row_get(row, "KIC", "KOI")
        name = str(int(kid)) if str(kid).replace(".", "").isdigit() else str(kid or "").strip()
        dra = _row_get(row, "_RA", "RAJ2000")
        dde = _row_get(row, "_DE", "DEJ2000")
        dr_css = _sep_arcsec(coord, dra, dde) if (_safe_float(dra) is not None and _safe_float(dde) is not None) else None
        out.append(
            CatalogMatch(
                catalog="CSS",
                name=name or "—",
                var_type=str(_row_get(row, "Type") or "").strip(),
                period=_safe_float(_row_get(row, "Per")),
                amplitude=_safe_float(_row_get(row, "Depth")),
                mag=_safe_float(_row_get(row, "Kpmag")),
                epoch=_safe_float(_row_get(row, "BJD")),
                delta_r=dr_css,
            )
        )
    return out


def _query_kelt(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    tl = _vizier_table(
        coord,
        radius_arcsec,
        "J/AJ/155/39/Variables",
        ["2MASS", "TIC", "Tmag", "Per-LS", "RAJ2000", "DEJ2000"],
    )
    if not tl or len(tl) == 0:
        return []
    t = tl[0]
    out: list[CatalogMatch] = []
    for row in t:
        tic = str(_row_get(row, "TIC") or "").strip()
        tm = str(_row_get(row, "2MASS") or "").strip()
        name = tic or tm or "—"
        out.append(
            CatalogMatch(
                catalog="KELT",
                name=name,
                var_type="",
                period=_safe_float(_row_get(row, "Per-LS")),
                mag=_safe_float(_row_get(row, "Tmag")),
                delta_r=_sep_arcsec(coord, _row_get(row, "RAJ2000"), _row_get(row, "DEJ2000")),
                extra={"2MASS": tm, "TIC": tic},
            )
        )
    return out


def _query_vsbs(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    tl = _vizier_table(
        coord,
        radius_arcsec,
        "J/A+A/598/A108/tablea11",
        ["Name", "Vmag", "Amp", "Per", "Type"],
    )
    if not tl or len(tl) == 0:
        return []
    t = tl[0]
    out: list[CatalogMatch] = []
    for row in t:
        out.append(
            CatalogMatch(
                catalog="VSBS",
                name=str(_row_get(row, "Name") or "").strip() or "—",
                var_type=str(_row_get(row, "Type") or "").strip(),
                period=_safe_float(_row_get(row, "Per")),
                amplitude=_safe_float(_row_get(row, "Amp")),
                mag=_safe_float(_row_get(row, "Vmag")),
                delta_r=_sep_arcsec(coord, _row_get(row, "RAJ2000"), _row_get(row, "DEJ2000")),
            )
        )
    return out


def _query_tess_eb(coord: SkyCoord, radius_arcsec: float) -> list[CatalogMatch]:
    tl = _vizier_table(
        coord,
        radius_arcsec,
        "J/ApJS/258/16/tess-ebs",
        ["TIC", "Tmag", "BJD0", "Per", "RAJ2000", "DEJ2000"],
    )
    if not tl or len(tl) == 0:
        return []
    t = tl[0]
    out: list[CatalogMatch] = []
    for row in t:
        out.append(
            CatalogMatch(
                catalog="TESS-EB",
                name=str(_row_get(row, "TIC") or "").strip() or "—",
                var_type="EB",
                period=_safe_float(_row_get(row, "Per")),
                mag=_safe_float(_row_get(row, "Tmag")),
                epoch=_safe_float(_row_get(row, "BJD0")),
                delta_r=_sep_arcsec(coord, _row_get(row, "RAJ2000"), _row_get(row, "DEJ2000")),
            )
        )
    return out


_CATALOG_WORKERS: list[tuple[str, Callable[[SkyCoord, float], list[CatalogMatch]]]] = [
    ("SIMBAD", _query_simbad),
    ("VSX", _query_vsx),
    ("ASAS-SN", _query_asassn),
    ("ZTF", _query_ztf),
    ("Gaia varisum", _query_gaia_varisum),
    ("ATLAS", _query_atlas),
    ("CSS", _query_css),
    ("KELT", _query_kelt),
    ("VSBS", _query_vsbs),
    ("TESS-EB", _query_tess_eb),
]


def check_candidate_in_catalogs(
    ra: float,
    dec: float,
    mag: float | None = None,
    radius_arcsec: float = 10.0,
) -> CrossmatchResult:
    coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
    matches: dict[str, list[CatalogMatch]] = {name: [] for name, _ in _CATALOG_WORKERS}
    errors: dict[str, str] = {}

    def _job(label: str, fn: Callable[[SkyCoord, float], list[CatalogMatch]]) -> tuple[str, list[CatalogMatch] | None, str | None]:
        try:
            return label, fn(coord, float(radius_arcsec)), None
        except Exception as exc:  # noqa: BLE001
            return label, None, str(exc)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_job, lab, fn): lab for lab, fn in _CATALOG_WORKERS}
        try:
            for fut in as_completed(futs, timeout=30.0):
                lab = futs[fut]
                try:
                    label, lst, err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors[lab] = str(exc)
                    continue
                if err:
                    errors[label] = err
                elif lst is not None:
                    matches[label] = lst
        except concurrent.futures.TimeoutError:
            for fut, lab in futs.items():
                if fut.done():
                    try:
                        label, lst, err = fut.result()
                        if err:
                            errors[label] = err
                        elif lst is not None:
                            matches[label] = lst
                    except Exception as exc:  # noqa: BLE001
                        errors[lab] = str(exc)
                else:
                    fut.cancel()
                    errors[lab] = "timeout (30s)"

    return CrossmatchResult(
        ra=float(ra),
        dec=float(dec),
        mag=mag,
        radius_arcsec=float(radius_arcsec),
        matches=matches,
        errors=errors,
    )


if __name__ == "__main__":
    # RR Lyr region (user test coordinates)
    ra, dec = 291.3543, 42.7847
    r = check_candidate_in_catalogs(ra, dec, mag=None, radius_arcsec=15.0)
    print("has_any_match:", r.has_any_match())
    print("best_period:", r.best_period())
    for cat, lst in r.matches.items():
        if lst:
            print(f"\n== {cat} ({len(lst)}) ==")
            for m in lst:
                print(" ", m.name, "|", m.summary(), "| mag", m.mag, "| dr", m.delta_r)
    for cat, msg in r.errors.items():
        print(f"\n!! {cat}: {msg}")
