#!/usr/bin/env python3
"""
VYVAR cross-validation HARNESS — reusable, any draft.

Consolidates the validated draft_000365 workflow:
  * independent photutils extraction (own Gaia DR3 catalogue from WCS, dual aperture,
    local re-centroid + annulus background),
  * independent sep / SExtractor extraction (mesh background + winpos + aperture),
  * decomposition against VYVAR per-frame dao_flux + reported lc_rms / comp_rms,
    using the SAME unweighted leave-one-out differential method.

Targets + comps are auto-loaded from VYVAR's comparison_stars_per_target.csv — nothing
is hardcoded. Shares only the input FITS with VYVAR.

REGRESSION TEST: run on draft_000365 first; it must reproduce
  target RMS ~0.171, sep comp RMS ~0.0105, photutils comp RMS ~0.0143.

Examples
--------
Full (FITS + VYVAR outputs reachable):
  python3 xval_run.py /home/milan/Public/VYVAR/draft_000365/detrended_aligned \
    --vyvar-photometry-dir .../draft_000365/platesolve/NoFilter_60_2/photometry \
    --proc-dir .../draft_000365/detrended_aligned/lights/NoFilter_60_2

FITS-only (photutils vs sep, no VYVAR comparison):
  python3 xval_run.py /home/milan/.../detrended_aligned

Deps: numpy pandas scipy astropy photutils astroquery matplotlib ; sep optional.
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=Warning)


# ── small helpers ─────────────────────────────────────────────────────────────
def log(m=""): print(m, flush=True)


from xval_harness_core import (
    assign_sep_confidence,
    comp_loo_median,
    diff_series,
    estimate_fwhm,
    find_frames,
    query_gaia,
    sclip_std,
)
from proc_frame_store import list_proc_csvs


def load_vyvar(photometry_dir: Path):
    """Return (summary_df, comps_long_df) with normalised string ids, or (None,None)."""
    sp = photometry_dir / "photometry_summary.csv"
    cp = photometry_dir / "comparison_stars_per_target.csv"
    if not sp.exists() or not cp.exists():
        return None, None
    s = pd.read_csv(sp, dtype={"catalog_id": str})
    c = pd.read_csv(cp, dtype={"catalog_id": str, "target_catalog_id": str})
    s["catalog_id"] = s["catalog_id"].astype(str).str.strip()
    c["catalog_id"] = c["catalog_id"].astype(str).str.strip()
    c["target_catalog_id"] = c["target_catalog_id"].astype(str).str.strip()
    return s, c


def load_dao(proc_dir: Path, ids: set[str]):
    files = list_proc_csvs(proc_dir)
    rows = []
    for fp in files:
        df = pd.read_csv(fp, dtype={"catalog_id": str})
        if "catalog_id" not in df.columns:
            continue
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        sub = df[df["catalog_id"].isin(ids)]
        if sub.empty:
            continue
        sf = sub["source_file"] if "source_file" in sub.columns else Path(fp).name
        for _, r in sub.iterrows():
            rows.append((r.get("source_file", Path(fp).name), r["catalog_id"], r["dao_flux"]))
    return pd.DataFrame(rows, columns=["frame", "source_id", "dao_flux"]) if rows else None


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detrended_aligned")
    ap.add_argument("--vyvar-photometry-dir", default=None)
    ap.add_argument("--proc-dir", default=None)
    ap.add_argument("--out", default="./tmp/xval_out")
    ap.add_argument("--gmax", type=float, default=16.0)
    ap.add_argument("--max-field", type=int, default=600)
    ap.add_argument("--delta-g-blend", type=float, default=2.5)
    args = ap.parse_args()

    root = Path(args.detrended_aligned).expanduser()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    proc_dir = Path(args.proc_dir).expanduser() if args.proc_dir else root
    PS = 9.768

    from astropy.io import fits as pyfits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.time import Time
    from astropy.stats import SigmaClip
    from photutils.aperture import (CircularAperture, CircularAnnulus,
                                    aperture_photometry, ApertureStats)
    from photutils.centroids import centroid_sources, centroid_com
    from scipy.spatial import cKDTree
    try:
        import sep; HAVE_SEP = True
    except ImportError:
        sep = None; HAVE_SEP = False
        log("note: sep not installed -> photutils-only (pip install sep --break-system-packages)")

    master, lights = find_frames(root)
    log(f"master {master and master.name} | lights {len(lights)}")
    with pyfits.open(master, memmap=False) as h:
        mhdr = h[0].header; mdata = np.ascontiguousarray(h[0].data, float)
    w = WCS(mhdr); ny, nx = mdata.shape
    cen = w.pixel_to_world(nx/2, ny/2)
    corners = w.pixel_to_world([0, nx, 0, nx], [0, 0, ny, ny])
    rdeg = float(cen.separation(corners).max().to(u.deg).value)

    # VYVAR targets + comps (drives what to measure)
    summ, comps_long = (None, None)
    if args.vyvar_photometry_dir:
        summ, comps_long = load_vyvar(Path(args.vyvar_photometry_dir).expanduser())
    have_vyvar = summ is not None
    log(f"VYVAR outputs: {'loaded' if have_vyvar else 'NOT provided -> photutils vs sep only'}")

    gdf = query_gaia(cen.ra.deg, cen.dec.deg, rdeg, args.gmax)
    gx, gy = w.world_to_pixel(SkyCoord(gdf.ra.values*u.deg, gdf.dec.values*u.deg))
    gdf["x"], gdf["y"] = gx, gy
    gdf = gdf[(gx > 40) & (gx < nx-40) & (gy > 40) & (gy < ny-40)].reset_index(drop=True)

    fwhm = estimate_fwhm(mdata, gdf.sort_values("phot_g_mean_mag").head(40)["x"].values,
                         gdf.sort_values("phot_g_mean_mag").head(40)["y"].values)
    r_ap = max(2.0*fwhm, 2.5); r_in, r_out = r_ap+3, r_ap+8; r_small = 3.0
    log(f"FWHM {fwhm:.2f}px | r_ap {r_ap:.2f} small {r_small} ann {r_in:.1f}-{r_out:.1f}")

    must = set()
    target_comps = {}
    if have_vyvar:
        for tid, grp in comps_long.groupby("target_catalog_id"):
            cs = [c for c in grp["catalog_id"].tolist() if c and c != "nan"]
            if len(cs) >= 3:
                target_comps[tid] = cs
                must |= {tid, *cs}
        log(f"VYVAR targets with >=3 comps: {len(target_comps)}; must-include ids: {len(must)}")

    # blend flag + selection (must-include always + brightest clean field)
    xy = gdf[["x", "y"]].values; G = gdf["phot_g_mean_mag"].values
    tree = cKDTree(xy); blended = np.zeros(len(gdf), bool)
    for i in range(len(gdf)):
        for j in tree.query_ball_point(xy[i], r_out):
            if j != i and G[j] < G[i] + args.delta_g_blend:
                blended[i] = True; break
    gdf["blended"] = blended
    gdf["must"] = gdf["source_id"].isin(must)
    mi = gdf[gdf["must"]]
    clean = gdf[(~gdf.blended) & (~gdf.must)].sort_values("phot_g_mean_mag").head(args.max_field)
    sel = pd.concat([mi, clean]).drop_duplicates("source_id").reset_index(drop=True)
    sel.to_csv(out / "xval_sources.csv", index=False)
    log(f"measuring {len(sel)} sources x {len(lights)} frames")

    # extraction
    x0, y0 = sel.x.values.astype(float), sel.y.values.astype(float)
    sids = sel.source_id.values; sc = SigmaClip(sigma=3.0)
    rows_p, rows_s, times = [], [], []
    sig = fwhm/2.3548
    for k, fp in enumerate(lights):
        with pyfits.open(fp, memmap=False) as h:
            hdr = h[0].header; d = np.ascontiguousarray(h[0].data, float)
        xc, yc = centroid_sources(d, x0, y0, box_size=9, centroid_func=centroid_com)
        bad = ~(np.isfinite(xc) & np.isfinite(yc)); xc[bad], yc[bad] = x0[bad], y0[bad]
        pos = np.column_stack([xc, yc])
        sky = ApertureStats(d, CircularAnnulus(pos, r_in, r_out), sigma_clip=sc).median
        fp_s = np.asarray(aperture_photometry(d, CircularAperture(pos, r_small))["aperture_sum"]) - sky*np.pi*r_small**2
        try: mjd = Time(hdr.get("DATE-OBS"), format="isot").mjd
        except Exception: mjd = np.nan
        times.append(mjd)
        for sid, v in zip(sids, fp_s): rows_p.append((fp.name, mjd, sid, float(v)))
        if HAVE_SEP:
            ds = d - sep.Background(d).back()
            xw, yw, _ = sep.winpos(ds, x0, y0, sig)
            fs, _, _ = sep.sum_circle(ds, xw, yw, r_small)
            for sid, v in zip(sids, fs): rows_s.append((fp.name, sid, float(v)))
        if (k+1) % 25 == 0 or k == len(lights)-1: log(f"  {k+1}/{len(lights)}")
    P = pd.DataFrame(rows_p, columns=["frame", "mjd", "source_id", "phot"])
    S = pd.DataFrame(rows_s, columns=["frame", "source_id", "sep"]) if HAVE_SEP else None

    wp = P.pivot_table(index="frame", columns="source_id", values="phot")
    ws = S.pivot_table(index="frame", columns="source_id", values="sep") if HAVE_SEP else None
    wd = None
    if have_vyvar:
        dao = load_dao(proc_dir, must)
        if dao is not None and len(dao):
            wd = dao.pivot_table(index="frame", columns="source_id", values="dao_flux")
            log(f"loaded VYVAR dao_flux: {dao['frame'].nunique()} frames x {dao['source_id'].nunique()} ids")

    # per-target decomposition
    if have_vyvar and target_comps:
        srows = []
        lc_map = dict(zip(summ["catalog_id"], summ.get("lc_rms", pd.Series())))
        vsx_map = dict(zip(summ["catalog_id"], summ.get("vsx_name", pd.Series())))
        for tid, cs in target_comps.items():
            t_p = sclip_std(diff_series(wp, tid, cs)) if tid in wp.columns else np.nan
            t_s = sclip_std(diff_series(ws, tid, cs)) if (HAVE_SEP and tid in ws.columns) else np.nan
            t_d = sclip_std(diff_series(wd, tid, cs)) if (wd is not None and tid in wd.columns) else np.nan
            cr_p = comp_loo_median(wp, cs)
            cr_s = comp_loo_median(ws, cs) if HAVE_SEP else np.nan
            cr_d = comp_loo_median(wd, cs) if wd is not None else np.nan
            srows.append((vsx_map.get(tid, ""), tid, lc_map.get(tid, np.nan),
                          t_p, t_s, t_d, cr_p, cr_s, cr_d))
        res = pd.DataFrame(srows, columns=[
            "vsx_name", "catalog_id", "vyvar_lc_rms",
            "target_rms_phot", "target_rms_sep", "target_rms_dao",
            "comp_rms_phot", "comp_rms_sep", "comp_rms_dao"])
        res["n_comp"] = res["catalog_id"].map(
            lambda cid: len(target_comps.get(str(cid).strip(), []))
        )
        res["confidence"] = [
            assign_sep_confidence(
                float(r.vyvar_lc_rms) if pd.notna(r.vyvar_lc_rms) else float("nan"),
                float(r.target_rms_sep) if pd.notna(r.target_rms_sep) else float("nan"),
                float(r.target_rms_dao) if pd.notna(r.target_rms_dao) else float("nan"),
                n_comp=int(r.n_comp) if pd.notna(r.n_comp) else 0,
            )
            for r in res.itertuples(index=False)
        ]
        res = res.sort_values("vyvar_lc_rms")
        res.to_csv(out / "xval_results.csv", index=False)
        log("\n=== PER-TARGET CROSS-VALIDATION ===")
        log(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        log(f"\nmedian target |phot-VYVAR|: "
            f"{(res.target_rms_phot - res.vyvar_lc_rms).abs().median():.4f} mag")
        if HAVE_SEP:
            log(f"median comp_rms  sep={res.comp_rms_sep.median():.4f}  "
                f"dao={res.comp_rms_dao.median():.4f}  phot={res.comp_rms_phot.median():.4f}")
        log(f"wrote {out/'xval_results.csv'}")
    else:
        log("\nphotutils vs sep only (no VYVAR comparison).")
        if HAVE_SEP:
            common = [s for s in wp.columns if s in ws.columns]
            log(f"common sources: {len(common)}  (per-target needs --vyvar-photometry-dir)")

    log("\nDONE.")


if __name__ == "__main__":
    raise SystemExit(main())
