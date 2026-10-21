"""GSPC zero point (GSPC-EXP-2026-10 -> GSPC-ZP-2026-10 -> GSPC-STD-2026-10).

Used by Phase 2A: second scale / note B for calibrated targets; the zero point (standard
export, ZP=GaiaDR3-GSPC) for note-A targets whose comps are outside the polynomial range.

Zero-point magnitudes of comparison stars from the Gaia DR3 Synthetic Photometry
Catalogue (GSPC; Gaia Collaboration, Montegriffo et al. 2023, A&A 674, A33) instead of the
Gaia G / BP-RP polynomial (Riello et al. 2021). Motivation: in dusty low-latitude fields
the polynomial is biased (draft 525 g: -25 to -63 mmag for BP-RP 1.0-2.0) and undefined
above BP-RP 2.0 for non-M-giants, while GSPC magnitudes come from the XP spectrum itself.

Online only: one ADQL query to the Gaia archive TAP service per field, cached as CSV in the
draft (no local database). Selection follows the paper (Sect. 6.2 and 7):

* flux / flux_error > 30 (the GSPC validity criterion, Eq. 21);
* |C*| < k x sigma_C*(G), sigma_C* from Riello et al. 2021 (Sect. 9.4), default k = 3;
* BP blended transits (xp_summary) below a fraction, default 10 %;
* ruwe < 1.4 optional (astrometric binaries; paper Sect. 7);
* X_flag: 1 = inside the validated G / BP-RP range, 0 = extrapolation. Policy "flag1" or
  "any".

Uncertainty per star: photon error of the synthetic flux plus a systematic floor
(default 10 mmag - SDSS red giants vs dwarfs and A0 <= 5 mag agree within ~10 mmag,
paper App. D / Sect. 7).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
CACHE_NAME = "gspc_cache.csv"
# VYVAR zero-point band -> GSPC column prefix
GSPC_BAND = {"g": "g_sdss", "r": "r_sdss", "i": "i_sdss", "V": "v_jkc", "R": "r_jkc", "B": "b_jkc", "I": "i_jkc"}
SN_MIN = 30.0
SIGMA_SYS_MAG = 0.010


def sigma_c_star(g_mag: float) -> float:
    """1-sigma scatter of C* for well-behaved isolated sources (Riello et al. 2021, Sect. 9.4, Eq. 18)."""
    return 0.0059898 + 8.817481e-12 * float(g_mag) ** 7.618399


def gspc_query(source_ids: Iterable[int | str], bands: Iterable[str] = ("g", "r", "i", "V", "R", "B", "I")) -> str:
    cols = []
    for b in bands:
        p = GSPC_BAND[b]
        cols += [f"x.{p}_mag", f"x.{p}_flux_error/x.{p}_flux AS {p}_relerr", f"x.{p}_flag"]
    ids = ",".join(str(s).strip() for s in source_ids if str(s).strip().isdigit())
    return (
        "SELECT s.source_id, s.phot_g_mean_mag AS g, s.bp_rp, s.ruwe, x.c_star, "
        + ", ".join(cols)
        + ", xs.bp_n_transits, xs.bp_n_blended_transits, xs.bp_n_contaminated_transits "
        "FROM gaiadr3.gaia_source AS s "
        "LEFT OUTER JOIN gaiadr3.synthetic_photometry_gspc AS x ON x.source_id = s.source_id "
        "LEFT OUTER JOIN gaiadr3.xp_summary AS xs ON xs.source_id = s.source_id "
        f"WHERE s.source_id IN ({ids})"
    )


def expected_columns(bands: Iterable[str] = ("g", "r", "i", "V", "R", "B", "I")) -> list[str]:
    cols = ["source_id", "g", "bp_rp", "ruwe", "c_star"]
    for b in bands:
        p = GSPC_BAND[b]
        cols += [f"{p}_mag", f"{p}_relerr", f"{p}_flag"]
    return cols + ["bp_n_transits", "bp_n_blended_transits", "bp_n_contaminated_transits"]


def _cache_header(path: Path) -> list[str]:
    try:
        with Path(path).open(newline="", encoding="utf-8") as f:
            return next(csv.reader(f), [])
    except (OSError, csv.Error):
        return []


def fetch_gspc(source_ids: Iterable[int | str], cache_dir: Path, *, timeout_s: float = 20.0,
               offline: bool = False) -> dict[str, dict[str, Any]]:
    """GSPC rows keyed by source_id. Reads ``cache_dir/gspc_cache.csv``; queries only the ids
    missing from it (online, Gaia archive TAP sync, POST) and appends them. A cache written
    with other columns (older query) is ignored and rebuilt, never appended to."""
    ids = [str(s).strip() for s in source_ids if str(s).strip().isdigit()]
    cache = Path(cache_dir) / CACHE_NAME
    stale = cache.is_file() and _cache_header(cache) != expected_columns()
    rows = {} if stale else read_cache(cache)
    missing = [s for s in ids if s not in rows]
    if missing and not offline:
        import urllib.parse  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        body = urllib.parse.urlencode(
            {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": gspc_query(missing)}
        ).encode()
        with urllib.request.urlopen(urllib.request.Request(TAP_SYNC_URL, data=body), timeout=timeout_s) as r:
            text = r.read().decode("utf-8")
        new = list(csv.DictReader(text.splitlines()))
        if new and "source_id" not in new[0]:
            raise ValueError(f"unexpected Gaia archive response: {text[:200]!r}")
        if new and list(new[0].keys()) != expected_columns():
            raise ValueError(f"unexpected Gaia archive columns: {list(new[0].keys())[:6]}...")
        if new:
            write_header = stale or not cache.is_file()
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("w" if write_header else "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=expected_columns())
                if write_header:
                    w.writeheader()
                w.writerows(new)
            for r in new:
                rows[str(r["source_id"])] = r
    return {s: rows[s] for s in ids if s in rows}


def read_cache(path: Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        with p.open(newline="", encoding="utf-8") as f:
            return {str(r["source_id"]): r for r in csv.DictReader(f) if r.get("source_id")}
    except (OSError, ValueError, KeyError, csv.Error):
        return {}


def _f(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def gspc_mag(row: dict[str, Any], band: str, *, flag_policy: str = "flag1", c_star_k: float = 3.0,
             max_blended_frac: float = 0.10, ruwe_max: float | None = None,
             sigma_sys: float = SIGMA_SYS_MAG) -> tuple[float, float, str]:
    """(magnitude, sigma, reason). NaN magnitude when the star fails a criterion; reason says
    which ('' when accepted)."""
    p = GSPC_BAND.get(band)
    if p is None:
        return float("nan"), float("nan"), f"band {band} not in GSPC"
    m = _f(row.get(f"{p}_mag"))
    rel = _f(row.get(f"{p}_relerr"))
    if not math.isfinite(m):
        return float("nan"), float("nan"), "no GSPC magnitude"
    if not (math.isfinite(rel) and rel > 0 and 1.0 / rel > SN_MIN):
        return float("nan"), float("nan"), "S/N <= 30"
    _fl = _f(row.get(f"{p}_flag"))
    flag = int(_fl) if math.isfinite(_fl) else 0
    if flag_policy == "flag1" and flag != 1:
        return float("nan"), float("nan"), "flag 0 (extrapolation)"
    g = _f(row.get("g"))
    cs = _f(row.get("c_star"))
    if math.isfinite(cs) and math.isfinite(g) and abs(cs) >= c_star_k * sigma_c_star(g):
        return float("nan"), float("nan"), "C* outlier"
    nt, nb = _f(row.get("bp_n_transits")), _f(row.get("bp_n_blended_transits"))
    if math.isfinite(nt) and nt > 0 and math.isfinite(nb) and nb / nt >= max_blended_frac:
        return float("nan"), float("nan"), "XP spectrum blended"
    if ruwe_max is not None and _f(row.get("ruwe")) >= ruwe_max:
        return float("nan"), float("nan"), "ruwe"
    sig = math.hypot(1.0857 * rel, sigma_sys)
    return m, sig, ""


#: GSPC-ZP-2026-10: a zero-point difference |C_GSPC - C_poly| at or above this flags the
#: standard calibration (note B). 2 x the ~10 mmag accuracy of standardised GSPC for red
#: giants and dwarfs (Montegriffo et al. 2023, App. D / Sect. 7).
ZP_NOTE_MIN_DIFF_MAG = 0.02
MIN_COMPS = 2


def gspc_zero_point(weights: dict[str, float], levels: dict[str, float],
                    rows: dict[str, dict[str, Any]], band: str) -> dict[str, Any]:
    """Zero point C_GSPC = sum_j w_j (g_j - M_j) / sum_j w_j over the comps with a valid GSPC
    magnitude (same estimator as photometry_lightcurve.ensemble_normalize; w = Broeg weights,
    M_j = Honeycutt level). Also the ensemble instrumental level sum w M / sum w over all comps.

    Returns {n_comp, n_eff, C, inst_level, sigma_model, sigma_data, rejected{reason: n}}; C is
    NaN with fewer than MIN_COMPS accepted comps."""
    out: dict[str, Any] = {"band": band, "n_comp": 0, "n_eff": None, "C": float("nan"),
                           "inst_level": float("nan"), "sigma_model": None, "sigma_data": None,
                           "rejected": {}}
    ids = [c for c in weights if math.isfinite(_f(levels.get(c))) and _f(weights[c]) > 0]
    if not ids:
        return out
    w_all = [float(weights[c]) for c in ids]
    out["inst_level"] = sum(w * _f(levels[c]) for w, c in zip(w_all, ids)) / sum(w_all)
    acc = []
    for c in ids:
        row = rows.get(str(c))
        if row is None:
            out["rejected"]["not in Gaia archive result"] = out["rejected"].get("not in Gaia archive result", 0) + 1
            continue
        m, s, why = gspc_mag(row, band)
        if why:
            out["rejected"][why] = out["rejected"].get(why, 0) + 1
            continue
        acc.append((float(weights[c]), m - _f(levels[c]), s))
        out.setdefault("accepted", []).append(str(c))
    out["n_comp"] = len(acc)
    if len(acc) < MIN_COMPS:
        return out
    sw = sum(a[0] for a in acc)
    d = sum(a[0] * a[1] for a in acc) / sw
    n_eff = sw * sw / sum(a[0] ** 2 for a in acc)
    out["C"] = d
    out["n_eff"] = round(n_eff, 3)
    out["sigma_model"] = round(math.sqrt(sum((a[0] * a[2]) ** 2 for a in acc)) / sw, 5)
    if len(acc) >= 3 and n_eff > 1.0:
        var = sum(a[0] * (a[1] - d) ** 2 for a in acc) / sw * n_eff / (n_eff - 1.0)
        out["sigma_data"] = round(math.sqrt(var) / math.sqrt(n_eff), 5)
    return out


def gspc_standard_available(gz: dict[str, Any] | None, lc_mag_gspc: Any = None) -> bool:
    """GSPC-STD-2026-10: a note-A target is exported as standard magnitudes on the GSPC scale
    when its GSPC zero point exists (>= MIN_COMPS accepted comps) and - when the LC column is
    given - ``mag_gspc`` has finite values. One rule for the export, the PDF and the note text."""
    gz = gz or {}
    if gz.get("C") is None or int(gz.get("n_comp") or 0) < MIN_COMPS:
        return False
    if lc_mag_gspc is None:
        return True
    try:
        import numpy as _np  # noqa: PLC0415

        v = _np.asarray(lc_mag_gspc, dtype=float)
        return bool(_np.isfinite(v).any())
    except (TypeError, ValueError):
        return False

