"""Hertzsprung–Russell diagram helpers from MASTERSTAR field catalog + local Gaia DR3 SQLite."""

from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id, normalize_gaia_source_id_series, read_vyvar_csv
from infolog import log_event

logger = logging.getLogger(__name__)


def _f(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return None
    try:
        x = float(val)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _gaia_table_name(conn: sqlite3.Connection) -> str:
    """Prefer ``gaia_dr3`` (VYVAR default) over ``gaia_source`` if both exist."""
    for t in ("gaia_dr3", "gaia_source"):
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
            (t,),
        ).fetchone()
        if row:
            return t
    return "gaia_dr3"


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info('{table}');")
    return {str(r[1]).strip().lower() for r in cur.fetchall()}


def _fetch_gaia_columns_by_source_id(
    gaia_db_path: Path,
    catalog_ids: list[str],
    want: list[str],
) -> pd.DataFrame:
    """Batch-fetch optional Gaia columns keyed by source_id (as str)."""
    if not gaia_db_path.is_file() or not want:
        return pd.DataFrame()
    ids_u: list[int] = []
    seen: set[int] = set()
    for raw in catalog_ids:
        s = str(raw or "").strip()
        if not s or not s.isdigit():
            continue
        try:
            sid = int(s)
        except (TypeError, ValueError, OverflowError):
            continue
        if sid in seen:
            continue
        seen.add(sid)
        ids_u.append(sid)
    if not ids_u:
        return pd.DataFrame()

    conn = sqlite3.connect(str(gaia_db_path))
    conn.row_factory = sqlite3.Row
    try:
        table = _gaia_table_name(conn)
        cols_db = _table_columns(conn, table)
        if "source_id" not in cols_db:
            return pd.DataFrame()

        parts: list[str] = ["source_id"]
        seen_sel: set[str] = {"source_id"}

        for w in want:
            wl = (w or "").strip().lower()
            if not wl or wl == "parallax_over_error":
                continue
            if wl in cols_db and wl not in seen_sel:
                parts.append(wl)
                seen_sel.add(wl)
                continue
            if wl == "phot_g_mean_mag" and "g_mag" in cols_db and "g_mag" not in seen_sel:
                parts.append("g_mag AS phot_g_mean_mag")
                seen_sel.add("g_mag")
                continue

        if "parallax_over_error" in {x.lower() for x in want}:
            if (
                "parallax" in cols_db
                and "parallax_error" in cols_db
                and "parallax_over_error" not in seen_sel
            ):
                parts.append("(parallax / NULLIF(parallax_error, 0)) AS parallax_over_error")
                seen_sel.add("parallax_over_error")

        sel = ", ".join(parts)
        out_rows: list[dict[str, Any]] = []
        bs = 500
        for i0 in range(0, len(ids_u), bs):
            chunk = ids_u[i0 : i0 + bs]
            ph = ",".join("?" * len(chunk))
            q = f"SELECT {sel} FROM {table} WHERE source_id IN ({ph});"
            for row in conn.execute(q, chunk):
                d = {k: row[k] for k in row.keys()}
                sid0 = d.get("source_id")
                if sid0 is None:
                    continue
                d["catalog_id"] = str(int(sid0))
                out_rows.append(d)
        if not out_rows:
            return pd.DataFrame()
        return pd.DataFrame(out_rows)
    finally:
        conn.close()


def _spectral_class(row: pd.Series) -> str:
    t = _f(row.get("teff_gspphot"))
    g = _f(row.get("logg_gspphot"))
    bprp = _f(row.get("bp_rp"))
    if t is None and bprp is None:
        return ""
    if t is not None:
        if t >= 30000:
            return "O–B"
        if t >= 10000:
            return "B–A"
        if t >= 7500:
            return "A–F"
        if t >= 6000:
            return "F–G"
        if t >= 5200:
            return "G"
        if t >= 3700:
            return "K"
        if t >= 2400:
            return "M"
        return "cool"
    if bprp is not None:
        if bprp < 0.0:
            return "hot (BP−RP)"
        if bprp < 0.5:
            return "early"
        if bprp < 1.2:
            return "solar-type"
        if bprp < 2.0:
            return "late"
        return "very cool"
    if g is not None and g < 3.5:
        return "evolved (log g)"
    return ""


def _classify_star(row: pd.Series) -> str:
    bp_rp = _f(row.get("bp_rp"))
    logg = _f(row.get("logg_gspphot"))
    abs_g = _f(row.get("abs_mag_g"))
    teff = _f(row.get("teff_gspphot"))
    nss = row.get("non_single_star", 0)

    if bp_rp is not None and bp_rp > 3.0:
        return "Carbon/Mira candidate"

    if bp_rp is not None and bp_rp > 1.5 and logg is not None and logg < 3.5:
        if logg < 1.5:
            return "Red supergiant"
        return "Red giant branch"

    # White dwarf — blue and faint (abs_mag > 10)
    if bp_rp is not None and bp_rp < 0.3 and abs_g is not None and abs_g > 10.0:
        return "White dwarf"

    if bp_rp is not None and bp_rp < -0.1 and abs_g is not None and abs_g < 0.0:
        return "WR / hot blue"

    try:
        if int(float(nss)) == 1:
            return "Binary candidate"
    except (TypeError, ValueError):
        pass
    _ = teff
    return ""


def build_hrd_dataframe(
    masterstars_csv: Path,
    gaia_db_path: Path,
    *,
    parallax_min_mas: float = 1.0,
    parallax_snr_min: float = 5.0,
) -> pd.DataFrame:
    """Build HRD dataframe from masterstars CSV + optional Gaia SQLite enrichment."""
    ms = read_vyvar_csv(masterstars_csv, low_memory=False)
    if ms.empty:
        return ms
    # Keep only stars with DAO detection (dao_flux or flux > 0)
    # dao_flux column name varies — try both
    _flux_col = None
    for _candidate in ("dao_flux", "flux", "peak_dao"):
        if _candidate in ms.columns:
            _numeric = pd.to_numeric(ms[_candidate], errors="coerce")
            if (_numeric > 0).any():
                _flux_col = _candidate
                break

    if _flux_col is not None:
        _before = len(ms)
        ms = ms[pd.to_numeric(ms[_flux_col], errors="coerce") > 0].copy()
        _after = len(ms)
        if _before != _after:
            log_event(
                f"HRD: filtered to DAO-detected stars only "
                f"({_after}/{_before} via {_flux_col})"
            )
    else:
        log_event("HRD: no flux column found — showing all masterstar rows")

    ms["catalog_id"] = normalize_gaia_source_id_series(ms["catalog_id"])
    if "phot_g_mean_mag" not in ms.columns and "g_mag" in ms.columns:
        ms["phot_g_mean_mag"] = pd.to_numeric(ms["g_mag"], errors="coerce")

    needed = ["teff_gspphot", "logg_gspphot", "parallax", "parallax_over_error", "non_single_star"]
    missing = [c for c in needed if c not in ms.columns]
    # CSV často obsahuje prázdne stĺpce z matchu — potom ``missing`` je prázdne a merge by zahodil
    # stĺpce z Gaia (starý dup_cols). Doplň fetch aj pre stĺpce bez jedinej platnej hodnoty.
    for c in ("teff_gspphot", "logg_gspphot", "non_single_star"):
        if c in ms.columns and c not in missing:
            num = pd.to_numeric(ms[c], errors="coerce")
            if int(num.notna().sum()) == 0:
                missing.append(c)
    if missing and gaia_db_path.is_file():
        gdf = _fetch_gaia_columns_by_source_id(
            Path(gaia_db_path),
            ms["catalog_id"].tolist(),
            missing + (["parallax_error"] if "parallax_over_error" in missing else []),
        )
        if not gdf.empty and "catalog_id" in gdf.columns:
            gdf = gdf.drop(columns=[c for c in ("source_id",) if c in gdf.columns], errors="ignore")
            # Povoliť prepis hodnôt z SQLite namiesto zahodenia duplicitných stĺpcov z gdf.
            overlap = [c for c in gdf.columns if c != "catalog_id" and c in ms.columns]
            if overlap:
                ms = ms.drop(columns=overlap, errors="ignore")
            ms = ms.merge(gdf, on="catalog_id", how="left", suffixes=("", "_gaia"))

    if "parallax_over_error" not in ms.columns and "parallax" in ms.columns and "parallax_error" in ms.columns:
        p = pd.to_numeric(ms["parallax"], errors="coerce")
        pe = pd.to_numeric(ms["parallax_error"], errors="coerce")
        ms["parallax_over_error"] = np.where((pe > 0) & np.isfinite(p) & np.isfinite(pe), p / pe, np.nan)

    p = pd.to_numeric(ms.get("parallax"), errors="coerce")
    snr = pd.to_numeric(ms.get("parallax_over_error"), errors="coerce")
    g = pd.to_numeric(ms.get("phot_g_mean_mag"), errors="coerce")
    ok = (
        np.isfinite(p)
        & np.isfinite(snr)
        & np.isfinite(g)
        & (p >= float(parallax_min_mas))
        & (snr >= float(parallax_snr_min))
    )
    dist_pc = np.where(ok, 1000.0 / p, np.nan)
    abs_mag = np.where(ok, g + 5.0 - 5.0 * np.log10(dist_pc), np.nan)
    ms["abs_mag_g"] = abs_mag
    ms["hrd_reliable"] = np.isfinite(abs_mag)
    ms["spectral_class"] = ms.apply(_spectral_class, axis=1)
    ms["interesting_label"] = ms.apply(_classify_star, axis=1)
    return ms


def _fmt(val: Any, spec: str) -> str:
    try:
        f = float(val)
        return format(f, spec) if math.isfinite(f) else "—"
    except (TypeError, ValueError):
        return "—"


def _make_row(row: pd.Series, category: str) -> dict[str, Any]:
    return {
        "catalog_id": str(row.get("catalog_id", "")),
        "category": category,
        "mag_g": _fmt(row.get("phot_g_mean_mag"), ".2f"),
        "abs_mag_g": _fmt(row.get("abs_mag_g"), ".2f"),
        "bp_rp": _fmt(row.get("bp_rp"), ".3f"),
        "teff": _fmt(row.get("teff_gspphot"), ".0f"),
        "logg": _fmt(row.get("logg_gspphot"), ".2f"),
        "ra_deg": _fmt(row.get("ra_deg"), ".4f"),
        "dec_deg": _fmt(row.get("dec_deg"), ".4f"),
        "x_px": _fmt(row.get("x"), ".0f"),
        "y_px": _fmt(row.get("y"), ".0f"),
    }


def get_top_interesting_stars(hrd_df: pd.DataFrame) -> pd.DataFrame:
    """Return highlighted stars for PDF/UI tables."""
    results: list[dict[str, Any]] = []
    if hrd_df is None or hrd_df.empty:
        return pd.DataFrame()
    df = hrd_df.copy()
    df["bp_rp"] = pd.to_numeric(df.get("bp_rp"), errors="coerce")
    df["abs_mag_g"] = pd.to_numeric(df.get("abs_mag_g"), errors="coerce")
    df["logg_gspphot"] = pd.to_numeric(df.get("logg_gspphot"), errors="coerce")
    if "non_single_star" in df.columns:
        nss = pd.to_numeric(df["non_single_star"], errors="coerce").fillna(0).astype(int)
    else:
        nss = pd.Series(0, index=df.index, dtype=int)

    red = df.nlargest(2, "bp_rp", keep="all")
    for _, r in red.iterrows():
        results.append(_make_row(r, "Reddest"))

    blue = df[df["bp_rp"].notna()].nsmallest(2, "bp_rp")
    for _, r in blue.iterrows():
        results.append(_make_row(r, "Bluest"))

    giants = df[(df["bp_rp"] > 1.5) & (df["logg_gspphot"] < 3.5) & df["logg_gspphot"].notna()].nlargest(2, "bp_rp")
    for _, r in giants.iterrows():
        results.append(_make_row(r, "Red giant branch"))

    wd = df[
        (df["bp_rp"] < 0.3) & (df["abs_mag_g"] > 10.0) & df["abs_mag_g"].notna()
    ].nsmallest(2, "bp_rp")
    for _, r in wd.iterrows():
        results.append(_make_row(r, "White dwarf"))

    carbon = df[df["bp_rp"] > 3.0].nlargest(2, "bp_rp")
    for _, r in carbon.iterrows():
        results.append(_make_row(r, "Carbon/Mira"))

    wr = df[(df["bp_rp"] < -0.1) & (df["abs_mag_g"] < 0.0) & df["abs_mag_g"].notna()].nsmallest(2, "abs_mag_g")
    for _, r in wr.iterrows():
        results.append(_make_row(r, "WR / hot blue"))

    binary = df[nss == 1].head(3)
    for _, r in binary.iterrows():
        results.append(_make_row(r, "Binary candidate"))

    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results).drop_duplicates(subset=["catalog_id"])
    return out


def plot_hrd_matplotlib(
    hrd_df: pd.DataFrame,
    top_stars: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> Path:
    """Render HRD scatter to PNG (matplotlib Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    reliable = hrd_df[hrd_df["hrd_reliable"] == True].copy()  # noqa: E712
    unreliable = hrd_df[hrd_df["hrd_reliable"] != True].copy()

    if not reliable.empty and reliable["bp_rp"].notna().any():
        bp_rp_vals = reliable["bp_rp"].fillna(1.0).clip(-0.5, 4.0)
        norm = plt.Normalize(-0.5, 4.0)
        colors = cm.RdYlBu_r(norm(bp_rp_vals.to_numpy(dtype=float)))
        ax.scatter(
            reliable["bp_rp"],
            reliable["abs_mag_g"],
            c=colors,
            s=8,
            alpha=0.6,
            linewidths=0,
            label=f"Reliable ({len(reliable)})",
            zorder=2,
        )

    unreliable_plot = unreliable.copy()
    if not unreliable_plot.empty:
        unreliable_plot["phot_g_mean_mag"] = pd.to_numeric(
            unreliable_plot.get("phot_g_mean_mag"), errors="coerce"
        )
        unreliable_plot["bp_rp"] = pd.to_numeric(unreliable_plot.get("bp_rp"), errors="coerce")
        unreliable_plot = unreliable_plot[
            np.isfinite(unreliable_plot["phot_g_mean_mag"]) & np.isfinite(unreliable_plot["bp_rp"])
        ]
    if not unreliable_plot.empty:
        ax.scatter(
            unreliable_plot["bp_rp"],
            unreliable_plot["phot_g_mean_mag"],
            c="gray",
            s=3,
            alpha=0.2,
            linewidths=0,
            label=f"No parallax ({len(unreliable_plot)})",
            zorder=1,
        )

    if top_stars is not None and not top_stars.empty:
        for _, star in top_stars.iterrows():
            cid = normalize_gaia_source_id(star.get("catalog_id", ""))
            match = hrd_df[hrd_df["catalog_id"] == cid]
            if match.empty:
                continue
            r = match.iloc[0]
            bprp = _f(r.get("bp_rp"))
            if bool(r.get("hrd_reliable")):
                y_val = _f(r.get("abs_mag_g"))
            else:
                y_val = _f(r.get("phot_g_mean_mag"))
            if bprp is None or y_val is None:
                continue
            ax.scatter(
                bprp,
                y_val,
                s=80,
                zorder=5,
                edgecolors="white",
                linewidths=0.8,
                facecolors="none",
            )
            lab = str(star.get("category", "")).strip()
            if len(lab) > 16:
                lab = lab[:14] + "\u2026"
            ax.annotate(
                lab,
                (bprp, y_val),
                fontsize=6,
                color="white",
                xytext=(5, 3),
                textcoords="offset points",
                zorder=6,
            )

    ax.invert_yaxis()
    ax.set_xlabel("BP − RP  [mag]", color="white", fontsize=11)
    ax.set_ylabel("M$_G$ / G  [mag]", color="white", fontsize=11)
    ax.set_title("Field Hertzsprung\u2013Russell diagram", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8, loc="upper left")

    sm = cm.ScalarMappable(norm=plt.Normalize(-0.5, 4.0), cmap=cm.RdYlBu_r)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("BP − RP", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    n_rel = int(hrd_df["hrd_reliable"].sum()) if "hrd_reliable" in hrd_df.columns else 0
    ax.text(
        0.02,
        0.02,
        f"N = {len(hrd_df)} stars  |  reliable: {n_rel}",
        transform=ax.transAxes,
        color="gray",
        fontsize=8,
    )
    plt.tight_layout()
    outp = Path(output_path) if output_path is not None else Path("hrd_field.png")
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outp), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outp


def resolve_clean_field_image_path(platesolve_dir: Path, photometry_dir: Path) -> Path | None:
    """Prefer clean masterstar PNGs; ``field_map*.png`` only as last resort (overlay squares)."""
    ps = Path(platesolve_dir)
    pt = Path(photometry_dir)
    for p in (ps / "masterstar_best.png", ps / "masterstar.png", pt.parent / "masterstar_best.png"):
        if p.is_file():
            return p
    for p in sorted(pt.parent.rglob("field_map*.png")):
        if p.is_file():
            return p
    return None


def _draft_dir_from_photometry(photometry_dir: Path) -> Path:
    """``.../Drafts/draft_XXX/platesolve/<obs>/photometry`` -> draft root."""
    return Path(photometry_dir).resolve().parent.parent.parent


def _obs_group_from_photometry(photometry_dir: Path) -> str:
    return str(Path(photometry_dir).resolve().parent.name)


def _masterstar_frame_score(df: pd.DataFrame) -> pd.Series:
    """Same spirit as ``photometry_report._compute_masterstar_score`` / UI dashboard (higher = better)."""
    score = pd.Series(0.0, index=df.index)
    if df.empty:
        return score

    def _norm_inverse(s: pd.Series) -> pd.Series:
        mn, mx = float(s.min()), float(s.max())
        if not (np.isfinite(mn) and np.isfinite(mx) and mx > mn):
            return pd.Series(1.0, index=s.index)
        return 1.0 - (s - mn) / (mx - mn)

    def _norm_direct(s: pd.Series) -> pd.Series:
        mn, mx = float(s.min()), float(s.max())
        if not (np.isfinite(mn) and np.isfinite(mx) and mx > mn):
            return pd.Series(1.0, index=s.index)
        return (s - mn) / (mx - mn)

    fwhm = pd.to_numeric(df.get("FWHM_PX"), errors="coerce")
    elong = pd.to_numeric(df.get("ELONGATION"), errors="coerce")
    stars = pd.to_numeric(df.get("STAR_COUNT"), errors="coerce")
    sky = pd.to_numeric(df.get("SKY_LEVEL"), errors="coerce")

    if fwhm.notna().sum() >= 2:
        score += 0.45 * _norm_inverse(fwhm.fillna(fwhm.max()))
    if elong.notna().sum() >= 2:
        score += 0.30 * _norm_inverse(elong.fillna(elong.max()))
    if stars.notna().sum() >= 2:
        score += 0.15 * _norm_direct(stars.fillna(stars.min()))
    if sky.notna().sum() >= 2:
        score += 0.10 * _norm_inverse(sky.fillna(sky.max()))
    return score


def fits_first_image_to_png(
    fits_path: Path,
    png_path: Path,
    *,
    lo_pct: float = 5.0,
    hi_pct: float = 99.5,
) -> bool:
    """Render primary HDU to 8-bit RGB PNG (percentile stretch)."""
    try:
        from astropy.io import fits
        from PIL import Image

        with fits.open(fits_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        if data.size == 0:
            return False
        ok = np.isfinite(data)
        if not ok.any():
            return False
        lo = float(np.nanpercentile(data[ok], float(lo_pct)))
        hi = float(np.nanpercentile(data[ok], float(hi_pct)))
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            lo = float(np.nanmin(data[ok]))
            hi = float(np.nanmax(data[ok]))
        if hi <= lo:
            hi = lo + 1e-6
        scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        gray = (scaled * 255.0).astype(np.uint8)
        gray[~ok] = 0
        img = Image.fromarray(gray, mode="L").convert("RGB")
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(png_path))
        return png_path.is_file()
    except Exception:  # noqa: BLE001
        logger.exception("fits_first_image_to_png failed for %s", fits_path)
        return False


def ensure_clean_field_background_png(
    platesolve_dir: Path,
    photometry_dir: Path,
    *,
    cache_dir: Path | None = None,
) -> Path | None:
    """
    Resolve a clean field snapshot for HRD annotations.

    Order: ``masterstar_best.png``, ``masterstar.png``, duplicate parent path,
    then any ``field_map*.png`` under the platesolve setup directory.
    If none exist, try top-3 frames from ``qc_metrics.csv`` (calibrated/ or processed/) as FITS→PNG.
    """
    hit = resolve_clean_field_image_path(platesolve_dir, photometry_dir)
    if hit is not None:
        return hit

    pt = Path(photometry_dir)
    cache = Path(cache_dir or (pt / "_hrd_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    out_png = cache / "hrd_field_from_fits.png"

    try:
        draft_dir = _draft_dir_from_photometry(pt)
        obs_group = _obs_group_from_photometry(pt)
        from pipeline import find_qc_metrics_csv

        qc_csv = find_qc_metrics_csv(draft_dir, app_config=None)
        if qc_csv is None:
            logger.info(
                "HRD: no clean PNG and no qc_metrics.csv under %s (calibrated/ or processed/)",
                draft_dir,
            )
            return None
        dfq = pd.read_csv(qc_csv, low_memory=False)
        if dfq.empty or "dst" not in dfq.columns:
            return None
        m = dfq["dst"].astype(str).str.contains(str(obs_group), regex=False)
        dfq = dfq.loc[m].copy()
        if dfq.empty:
            return None
        dfq["FWHM_PX"] = pd.to_numeric(dfq.get("fwhm_px"), errors="coerce")
        dfq["ELONGATION"] = pd.to_numeric(dfq.get("elongation"), errors="coerce")
        if "n_stars_detected" in dfq.columns:
            dfq["STAR_COUNT"] = pd.to_numeric(dfq.get("n_stars_detected"), errors="coerce")
        else:
            dfq["STAR_COUNT"] = pd.to_numeric(dfq.get("n_sources"), errors="coerce")
        dfq["SKY_LEVEL"] = pd.to_numeric(dfq.get("bg_median"), errors="coerce")
        dfq["_fits_path"] = dfq["dst"].map(lambda s: Path(str(s).strip()))
        dfq["_score"] = _masterstar_frame_score(dfq)
        best = dfq.sort_values("_score", ascending=False).head(3)
        for _, row in best.iterrows():
            fp = row["_fits_path"]
            if isinstance(fp, Path) and fp.is_file() and fits_first_image_to_png(fp, out_png):
                return out_png
    except Exception:  # noqa: BLE001
        logger.exception("ensure_clean_field_background_png failed")

    return None


def annotate_field_image(
    field_image_path: str | Path,
    top_stars: pd.DataFrame,
    hrd_df: pd.DataFrame,
) -> Path:
    """Mark interesting stars on a field PNG/JPEG."""
    from PIL import Image, ImageDraw

    field_image_path = Path(field_image_path)
    img = Image.open(str(field_image_path)).convert("RGB")
    draw = ImageDraw.Draw(img)

    category_colors: dict[str, tuple[int, int, int]] = {
        "Reddest": (255, 80, 80),
        "Bluest": (80, 80, 255),
        "Red giant branch": (255, 140, 0),
        "White dwarf": (220, 220, 255),
        "Carbon/Mira": (180, 0, 0),
        "WR / hot blue": (0, 200, 255),
        "Binary candidate": (0, 255, 100),
        "Carbon/Mira candidate": (180, 0, 0),
        "Red supergiant": (255, 100, 0),
    }

    w, h = img.size
    if top_stars is None or top_stars.empty:
        out_path = field_image_path.parent / "hrd_field_annotated.png"
        img.save(str(out_path))
        return out_path

    for _, star in top_stars.iterrows():
        cid = normalize_gaia_source_id(star.get("catalog_id", ""))
        match = hrd_df[hrd_df["catalog_id"] == cid]
        if match.empty:
            continue
        r = match.iloc[0]
        x = _f(r.get("x"))
        y = _f(r.get("y"))
        if x is None or y is None:
            continue
        if not (10 < x < w - 10 and 10 < y < h - 10):
            continue
        cat_raw = str(star.get("category", "")).strip()
        color = category_colors.get(cat_raw)
        if color is None:
            pref = cat_raw.split("/")[0].strip()
            color = category_colors.get(pref, (255, 255, 0))
        radius = 18
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=color,
            width=2,
        )
        label = cat_raw.split()[0] if cat_raw else "?"
        draw.text((x + radius + 3, y - 8), label, fill=color)

    out_path = field_image_path.parent / "hrd_field_annotated.png"
    img.save(str(out_path))
    return out_path
