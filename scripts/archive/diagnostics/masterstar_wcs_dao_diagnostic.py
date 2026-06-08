"""DAO vs Gaia→pixel reziduály pre MASTERSTAR (použitie z UI aj z CLI)."""

from __future__ import annotations

import math
from io import StringIO
from pathlib import Path
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd

from gaia_catalog_id import masterstar_row_gaia_key, normalize_gaia_source_id


def resolve_paths_from_archive(archive_root: Path | str) -> tuple[Path, Path, Path | None]:
    """Vráti (MASTERSTAR.fits, masterstars csv, field_catalog_cone alebo None)."""
    root = Path(archive_root)
    ps = root / "platesolve"
    fits_path = ps / "MASTERSTAR.fits"
    cand = ps / "masterstars_full_match.csv"
    csv_path = cand if cand.is_file() else ps / "masterstars.csv"
    cone_path = ps / "field_catalog_cone.csv"
    if not cone_path.is_file():
        cone_path = None
    return fits_path, csv_path, cone_path


def _load_wcs_from_fits(fits_path: Path):
    from astropy.io import fits
    from astropy.wcs import FITSFixedWarning, WCS

    with fits.open(fits_path, memmap=False) as hdul:
        hdr = hdul[0].header
    with catch_warnings():
        simplefilter("ignore", FITSFixedWarning)
        w = WCS(hdr)
    if not getattr(w, "has_celestial", False):
        raise ValueError("FITS nemá použiteľný celestný WCS (has_celestial=False).")
    return w, hdr


def _world_to_pix_xy(w, ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra = np.asarray(ra_deg, dtype=np.float64).ravel()
    de = np.asarray(dec_deg, dtype=np.float64).ravel()
    m = np.isfinite(ra) & np.isfinite(de)
    out_x = np.full(ra.shape, np.nan, dtype=np.float64)
    out_y = np.full(ra.shape, np.nan, dtype=np.float64)
    if not np.any(m):
        return out_x, out_y
    coo = SkyCoord(ra=ra[m] * u.deg, dec=de[m] * u.deg, frame="icrs")
    px, py = w.world_to_pixel(coo)
    out_x[m] = np.asarray(px, dtype=np.float64).ravel()
    out_y[m] = np.asarray(py, dtype=np.float64).ravel()
    return out_x, out_y


def _stats_lines(dx: np.ndarray, dy: np.ndarray, label: str) -> list[str]:
    lines: list[str] = []
    ok = np.isfinite(dx) & np.isfinite(dy)
    if not np.any(ok):
        lines.append(f"{label}: žiadne platné páry.")
        return lines
    ddx = dx[ok]
    ddy = dy[ok]
    dr = np.hypot(ddx, ddy)
    lines.append(f"\n=== {label} (N={int(np.sum(ok))}) ===")
    lines.append(
        f"  dx [px]: median={np.median(ddx):+.4f}  MAD≈{np.median(np.abs(ddx - np.median(ddx))) * 1.4826:.4f}"
    )
    lines.append(
        f"  dy [px]: median={np.median(ddy):+.4f}  MAD≈{np.median(np.abs(ddy - np.median(ddy))) * 1.4826:.4f}"
    )
    lines.append(f"  |dr| [px]: median={np.median(dr):.4f}  RMS={float(np.sqrt(np.mean(dr**2))):.4f}")
    lines.append(
        f"  percentily |dr|: p50={np.percentile(dr, 50):.3f}  p90={np.percentile(dr, 90):.3f} "
        f"p99={np.percentile(dr, 99):.3f}  max={np.max(dr):.3f}"
    )
    return lines


def _worst_table_lines(ms: pd.DataFrame, cols: list[str], n: int) -> list[str]:
    lines: list[str] = []
    if n <= 0 or ms.empty or "dr" not in ms.columns:
        return lines
    sub = ms[cols].copy().sort_values("dr", ascending=False).head(n)
    lines.append(f"\n--- Top {len(sub)} najhorších (podľa |dr|) ---")
    buf = StringIO()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        buf.write(sub.to_string(index=False))
    lines.append(buf.getvalue())
    return lines


def run_masterstar_wcs_dao_diagnostic(
    fits_path: Path | str,
    csv_path: Path | str,
    *,
    cone_path: Path | str | None = None,
    worst_n: int = 12,
) -> str:
    """Vráti viacriadkový textový report (round-trip WCS + DAO vs Gaia→pixel ak je kužeľ)."""
    fits_path = Path(fits_path)
    csv_path = Path(csv_path)
    cone_p: Path | None = Path(cone_path) if cone_path else None
    if cone_p is not None and not cone_p.is_file():
        cone_p = None

    if not fits_path.is_file():
        raise ValueError(f"Chýba FITS: {fits_path}")
    if not csv_path.is_file():
        raise ValueError(f"Chýba CSV: {csv_path}")

    out: list[str] = []

    w, _hdr = _load_wcs_from_fits(fits_path)
    ms = pd.read_csv(csv_path)
    need = {"x", "y", "ra_deg", "dec_deg"}
    if not need.issubset(set(ms.columns)):
        raise ValueError(f"V CSV chýbajú stĺpce {need - set(ms.columns)}.")

    x_dao = pd.to_numeric(ms["x"], errors="coerce").to_numpy(dtype=np.float64)
    y_dao = pd.to_numeric(ms["y"], errors="coerce").to_numpy(dtype=np.float64)
    ra_csv = pd.to_numeric(ms["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de_csv = pd.to_numeric(ms["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)

    x_rt, y_rt = _world_to_pix_xy(w, ra_csv, de_csv)
    dx_rt = x_dao - x_rt
    dy_rt = y_dao - y_rt
    out.extend(_stats_lines(dx_rt, dy_rt, "Round-trip: DAO(x,y) − world_to_pixel(CSV ra_deg/dec_deg z DAO)"))

    med_rt = float(np.median(np.hypot(dx_rt, dy_rt)[np.isfinite(dx_rt) & np.isfinite(dy_rt)]))

    if "catalog_id" not in ms.columns:
        out.append("\n(Stĺpec catalog_id chýba — preskočená Gaia→pixel časť.)")
        out.append("\n--- Stručný záver ---")
        if math.isfinite(med_rt) and med_rt > 0.5:
            out.append(
                f"Round-trip median |dr|={med_rt:.2f} px — WCS sa môže líšiť od stavu pri exporte CSV alebo SIP/konvencia."
            )
        elif math.isfinite(med_rt):
            out.append(f"Round-trip median |dr|={med_rt:.3f} px — WCS je interne konzistentný s CSV ra/dec (DAO).")
        out.append(f"\nFITS: {fits_path.resolve()}")
        out.append(f"CSV:  {csv_path.resolve()}")
        out.append(f"Cone: {str(cone_p.resolve()) if cone_p is not None else '(n/a)'}")
        return "\n".join(out)

    cid = ms["catalog_id"].fillna("").astype(str).str.strip()
    matched = cid.ne("")
    n_m = int(matched.sum())
    out.append(f"\nRiadkov s catalog_id: {n_m} / {len(ms)}")

    med_dr = float("nan")
    med_dr_tight_2as = float("nan")

    if cone_p is None:
        out.append(
            "Bez field_catalog_cone.csv nie sú k dispozícii skutočné Gaia RA/Dec z kužela — "
            "spusti pipeline s exportom kužela alebo zadaj cestu v CLI (--cone)."
        )
    else:
        fc = pd.read_csv(cone_p)
        if "catalog_id" not in fc.columns or "ra_deg" not in fc.columns or "dec_deg" not in fc.columns:
            raise ValueError("field_catalog_cone.csv očakáva stĺpce catalog_id, ra_deg, dec_deg.")

        fc = fc.copy()
        fc["_k"] = fc["catalog_id"].map(normalize_gaia_source_id)
        fc = fc[fc["_k"].ne("")].drop_duplicates(subset=["_k"], keep="first")
        cmap = fc.set_index("_k")[["ra_deg", "dec_deg"]]

        sub = ms.loc[matched].copy()
        sub["_k"] = sub.apply(masterstar_row_gaia_key, axis=1)
        hit = sub["_k"].isin(cmap.index)
        n_hit = int(hit.sum())
        out.append(
            f"Z toho nájdených v field_catalog_cone.csv: {n_hit} "
            f"(join kľúč: normalizovaný catalog_id, príp. stĺpec name ak je číselný Gaia ID)"
        )

        if n_hit == 0:
            out.append("Žiadny prienik catalog_id — skontroluj, či kužeľ patrí k tomu istému behu pipeline.")
        else:
            sub = sub.loc[hit]
            _keys = sub["_k"].to_numpy(dtype=object)
            ra_g = pd.to_numeric(cmap.reindex(_keys)["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
            de_g = pd.to_numeric(cmap.reindex(_keys)["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
            x_g, y_g = _world_to_pix_xy(w, ra_g, de_g)
            xd = pd.to_numeric(sub["x"], errors="coerce").to_numpy(dtype=np.float64)
            yd = pd.to_numeric(sub["y"], errors="coerce").to_numpy(dtype=np.float64)
            dx_g = xd - x_g
            dy_g = yd - y_g
            out.extend(_stats_lines(dx_g, dy_g, "DAO − world_to_pixel(Gaia RA/Dec z field_catalog_cone)"))

            dr = np.hypot(dx_g, dy_g)
            med_dr = float(np.median(dr[np.isfinite(dr)])) if np.any(np.isfinite(dr)) else float("nan")
            sub = sub.assign(
                dx_gaia_px=dx_g,
                dy_gaia_px=dy_g,
                dr=dr,
                x_gaia_px=x_g,
                y_gaia_px=y_g,
            )
            worst_cols = [
                "name",
                "catalog_id",
                "x",
                "y",
                "x_gaia_px",
                "y_gaia_px",
                "dx_gaia_px",
                "dy_gaia_px",
                "dr",
            ]
            if "match_sep_arcsec" in sub.columns:
                worst_cols.append("match_sep_arcsec")
            out.extend(_worst_table_lines(sub, worst_cols, int(worst_n)))

            if "match_sep_arcsec" in sub.columns:
                sep = pd.to_numeric(sub["match_sep_arcsec"], errors="coerce").to_numpy(dtype=np.float64)
                out.append(
                    "\n(Poznámka: riadky s veľkým match_sep_arcsec sú často „zlé páry“ voči Gaia — "
                    "umelo zväčšujú median |dr| nižšie.)"
                )
                for thr_arc, label in ((1.0, "≤1.0″"), (2.0, "≤2.0″"), (5.0, "≤5.0″")):
                    m = np.isfinite(dx_g) & np.isfinite(dy_g) & np.isfinite(sep) & (sep <= thr_arc)
                    n_sub = int(np.count_nonzero(m))
                    if n_sub >= 8:
                        out.extend(
                            _stats_lines(
                                dx_g[m],
                                dy_g[m],
                                f"DAO − Gaia→pixel (iba match_sep_arcsec {label}, N={n_sub})",
                            )
                        )
                        if thr_arc == 2.0:
                            dr_t = np.hypot(dx_g[m], dy_g[m])
                            med_dr_tight_2as = float(np.median(dr_t)) if np.any(np.isfinite(dr_t)) else float("nan")
                if math.isfinite(med_dr_tight_2as):
                    out.append(
                        f"\nPre zhody ≤2″: median |dr|={med_dr_tight_2as:.3f} px "
                        f"(lepší indikátor WCS než zmiešaná množina so širokými separáciami)."
                    )

    out.append("\n--- Stručný záver ---")
    if math.isfinite(med_rt) and med_rt > 0.5:
        out.append(
            f"Round-trip median |dr|={med_rt:.2f} px — WCS v súbore sa pravdepodobne líši od stavu pri exporte CSV, "
            "alebo je problém so SIP/konvenciou pixlov."
        )
    elif math.isfinite(med_rt):
        out.append(f"Round-trip median |dr|={med_rt:.3f} px — WCS je interne konzistentný s CSV ra/dec (DAO).")

    if math.isfinite(med_dr):
        if med_dr > 2.0:
            out.append(
                f"Gaia→pixel (všetky spárované v joini) median |dr|={med_dr:.2f} px — "
                "pri veľkom match_sep_arcsec ide často o zlé páry, nie čisto o WCS; pozri filtrované bloky vyššie."
            )
        else:
            out.append(
                f"Gaia→pixel median |dr|={med_dr:.2f} px — pod ~2 px často akceptovateľné; over vizuálne na jasných hviezdach."
            )
        if math.isfinite(med_dr_tight_2as) and med_dr > med_dr_tight_2as + 0.5:
            out.append(
                f"Pre úzke zhody (≤2″) bol median |dr|≈{med_dr_tight_2as:.2f} px oproti {med_dr:.2f} px celkom — "
                "globálna hodnota bola zmazaná širokými zhodami."
            )
    elif cone_p is not None:
        out.append("Gaia→pixel: bez platných párov pre súhrn median |dr|.")

    out.append(f"\nFITS: {fits_path.resolve()}")
    out.append(f"CSV:  {csv_path.resolve()}")
    out.append(f"Cone: {cone_p.resolve() if cone_p is not None else '(n/a)'}")
    return "\n".join(out)
