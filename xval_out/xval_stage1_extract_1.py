#!/usr/bin/env python3
"""
VYVAR cross-validation — STAGE 1: independent extraction (FITS-only)

What it does (fully independent of VYVAR's intermediate products):
  1) Reads the MASTERSTAR (deep reference) for WCS + FWHM + source detection ref.
  2) Queries Gaia DR3 around the WCS field centre -> an INDEPENDENT star list.
  3) Projects Gaia -> pixels, keeps on-chip + isolated sources, caps to the
     brightest MAX_SOURCES (target + comps + a magnitude spread for RMS-vs-mag).
  4) Measures FWHM on the masterstar to set the aperture.
  5) For every aligned light frame: local re-centroid each star, then circular
     aperture photometry with a local sky annulus (photutils 1.13).
  6) Saves a flux matrix + source table + run metadata, and prints diagnostics.

Independence: shares ONLY the input FITS with VYVAR. Own catalogue, own
detection, own apertures, own background. No VYVAR CSVs are read.

Read-only on inputs. Writes only into OUT_DIR (default ./xval_out).

Usage:
    python3 xval_stage1_extract.py /home/milan/Public/VYVAR/draft_000365/detrended_aligned

Paste the whole printed output back into the chat. Then I'll write Stage 2
(differential LCs + comparison to VYVAR) tuned to what Stage 1 reports.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=Warning)  # quiet FITS/WCS fixups

# ── Tunables (sensible defaults; I'll revise after seeing Stage 1 output) ──────
GMAX = 16.0            # Gaia G faint cut for the source list
MAX_SOURCES = 350      # keep brightest N on-chip isolated sources
CHIP_MARGIN_PX = 40    # drop sources within this many px of any edge
ISO_FACTOR = 3.0       # require nearest neighbour > ISO_FACTOR * r_ap
CENTROID_BOX = 9       # px box for per-frame local re-centroid
FWHM_NSTARS = 40       # bright isolated stars used to estimate FWHM
R_AP_FWHM = 2.0        # aperture radius = R_AP_FWHM * FWHM
R_AP_MIN_PX = 2.5      # floor on aperture radius
ANN_IN_PAD = 3.0       # sky annulus inner = r_ap + this (px)
ANN_OUT_PAD = 8.0      # sky annulus outer = r_ap + this (px)
OUT_DIR = Path("./xval_out")
# ──────────────────────────────────────────────────────────────────────────────


def log(msg: str = "") -> None:
    print(msg, flush=True)


def find_light_frames(root: Path) -> tuple[Path | None, list[Path]]:
    pats = ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS")
    files: list[Path] = []
    for p in pats:
        files += [Path(x) for x in glob.glob(str(root / "**" / p), recursive=True)]
    files = sorted(set(files))
    master = None
    lights: list[Path] = []
    for f in files:
        if "MASTER" in f.name.upper():
            master = f if master is None else master
        else:
            lights.append(f)
    return master, lights


def estimate_fwhm(data: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                  box: int = 11) -> float:
    """Median FWHM from flux-weighted 2nd moments on bright stamps."""
    half = box // 2
    fwhms: list[float] = []
    ny, nx = data.shape
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half >= nx or yi + half >= ny:
            continue
        stamp = data[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(float)
        bkg = np.median(stamp)
        s = stamp - bkg
        s[s < 0] = 0.0
        tot = s.sum()
        if tot <= 0:
            continue
        yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
        cx = (s * xx).sum() / tot
        cy = (s * yy).sum() / tot
        sx2 = (s * (xx - cx) ** 2).sum() / tot
        sy2 = (s * (yy - cy) ** 2).sum() / tot
        if sx2 <= 0 or sy2 <= 0:
            continue
        sigma = np.sqrt(np.sqrt(sx2 * sy2))
        fwhms.append(2.3548 * sigma)
    if not fwhms:
        return float("nan")
    return float(np.median(fwhms))


def query_gaia(ra0: float, dec0: float, radius_deg: float, gmax: float) -> pd.DataFrame:
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = 200000
    adql = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                         CIRCLE('ICRS', {ra0}, {dec0}, {radius_deg}))
          AND phot_g_mean_mag IS NOT NULL
          AND phot_g_mean_mag < {gmax}
    """
    job = Gaia.launch_job_async(adql)
    tbl = job.get_results()
    df = tbl.to_pandas()
    # astroquery/TAP can vary column case (SOURCE_ID is often uppercase) -> normalise
    df.columns = [str(c).lower() for c in df.columns]
    print(f"  Gaia returned columns: {list(df.columns)}")

    def pick(cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        return None

    colmap = {
        "source_id": pick(["source_id", "dr3_source_id"]),
        "ra": pick(["ra", "ra_icrs"]),
        "dec": pick(["dec", "de", "dec_icrs"]),
        "phot_g_mean_mag": pick(["phot_g_mean_mag", "gmag", "g"]),
        "bp_rp": pick(["bp_rp", "bprp"]),
    }
    missing = [k for k, v in colmap.items() if v is None]
    if missing:
        raise KeyError(f"Gaia result missing {missing}; available: {list(df.columns)}")
    out = df[[colmap["source_id"], colmap["ra"], colmap["dec"],
              colmap["phot_g_mean_mag"], colmap["bp_rp"]]].copy()
    out.columns = ["source_id", "ra", "dec", "phot_g_mean_mag", "bp_rp"]
    return out


def main() -> int:
    t_start = datetime.now(timezone.utc)
    root = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.cwd()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from astropy.io import fits as pyfits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.time import Time
    from photutils.aperture import (CircularAperture, CircularAnnulus,
                                    aperture_photometry, ApertureStats)
    from photutils.centroids import centroid_sources, centroid_com
    from astropy.stats import SigmaClip

    log("=" * 72)
    log("STAGE 1 — independent extraction")
    log("=" * 72)

    master, lights = find_light_frames(root)
    log(f"root              : {root}")
    log(f"masterstar        : {master}")
    log(f"light frames      : {len(lights)} (MASTERSTAR excluded)")
    if master is None or not lights:
        log("!! need a MASTERSTAR + light frames; check the path.")
        return 1

    # ── reference frame: WCS + FWHM ──────────────────────────────────────────
    with pyfits.open(master, memmap=False) as hdul:
        mhdr = hdul[0].header
        mdata = np.ascontiguousarray(hdul[0].data, dtype=np.float64)
    w = WCS(mhdr)
    ny, nx = mdata.shape

    # field centre + covering radius from the 4 corners
    cen = w.pixel_to_world(nx / 2.0, ny / 2.0)
    corners = w.pixel_to_world([0, nx, 0, nx], [0, 0, ny, ny])
    radius_deg = float(cen.separation(corners).max().to(u.deg).value)
    log(f"field centre      : RA={cen.ra.deg:.5f}  Dec={cen.dec.deg:.5f}")
    log(f"covering radius   : {radius_deg:.3f} deg")
    log(f"OBJECT header     : {mhdr.get('OBJECT', '-')}  "
        f"(trust WCS over this keyword on this rig)")

    # ── Gaia independent catalogue ───────────────────────────────────────────
    log("-" * 72)
    log(f"Querying Gaia DR3 (G < {GMAX}) ...")
    gdf = query_gaia(cen.ra.deg, cen.dec.deg, radius_deg, GMAX)
    log(f"Gaia rows returned: {len(gdf)}")

    sky = SkyCoord(ra=gdf["ra"].values * u.deg, dec=gdf["dec"].values * u.deg)
    gx, gy = w.world_to_pixel(sky)
    gdf["x"] = gx
    gdf["y"] = gy
    on = ((gx > CHIP_MARGIN_PX) & (gx < nx - CHIP_MARGIN_PX)
          & (gy > CHIP_MARGIN_PX) & (gy < ny - CHIP_MARGIN_PX))
    gdf = gdf[on].reset_index(drop=True)
    log(f"on-chip (margin {CHIP_MARGIN_PX}px): {len(gdf)}")

    # FWHM from bright on-chip stars
    bright = gdf.sort_values("phot_g_mean_mag").head(FWHM_NSTARS)
    fwhm = estimate_fwhm(mdata, bright["x"].values, bright["y"].values)
    r_ap = max(R_AP_FWHM * fwhm, R_AP_MIN_PX) if np.isfinite(fwhm) else R_AP_MIN_PX
    r_in, r_out = r_ap + ANN_IN_PAD, r_ap + ANN_OUT_PAD
    log("-" * 72)
    log(f"measured FWHM     : {fwhm:.2f} px  ({fwhm * 9.768:.1f} arcsec)")
    log(f"aperture r_ap     : {r_ap:.2f} px   annulus: {r_in:.2f}-{r_out:.2f} px")

    # isolation filter (nearest-neighbour in px)
    xy = gdf[["x", "y"]].values
    iso = np.full(len(gdf), np.inf)
    for i in range(len(gdf)):
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        d[i] = np.inf
        iso[i] = d.min()
    gdf["nn_px"] = iso
    iso_min = ISO_FACTOR * r_ap
    keep = gdf["nn_px"] > iso_min
    n_blend = int((~keep).sum())
    gdf = gdf[keep].sort_values("phot_g_mean_mag").head(MAX_SOURCES).reset_index(drop=True)
    log(f"isolation cut (nn > {iso_min:.1f}px): dropped {n_blend} blended; "
        f"kept {len(gdf)} (capped at {MAX_SOURCES})")
    log(f"G range kept      : {gdf['phot_g_mean_mag'].min():.2f} .. "
        f"{gdf['phot_g_mean_mag'].max():.2f}")

    src_path = OUT_DIR / "xval_sources.csv"
    gdf.to_csv(src_path, index=False)
    log(f"wrote source table: {src_path}")

    # ── per-frame aperture photometry ────────────────────────────────────────
    log("-" * 72)
    log(f"Measuring {len(gdf)} sources on {len(lights)} frames ...")
    x0 = gdf["x"].values.astype(float)
    y0 = gdf["y"].values.astype(float)
    sids = gdf["source_id"].values
    sclip = SigmaClip(sigma=3.0)

    rows = []           # long-format flux matrix
    drift_med = []      # per-frame median recentroid offset (alignment check)
    times = []
    for k, fpath in enumerate(lights):
        try:
            with pyfits.open(fpath, memmap=False) as hdul:
                fhdr = hdul[0].header
                fdata = np.ascontiguousarray(hdul[0].data, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            log(f"  [skip] {fpath.name}: read error {exc}")
            continue

        # local re-centroid (also measures alignment residual)
        xc, yc = centroid_sources(fdata, x0, y0, box_size=CENTROID_BOX,
                                  centroid_func=centroid_com)
        bad = ~(np.isfinite(xc) & np.isfinite(yc))
        xc[bad], yc[bad] = x0[bad], y0[bad]
        drift = np.hypot(xc - x0, yc - y0)
        drift_med.append(float(np.nanmedian(drift)))

        pos = np.column_stack([xc, yc])
        ap = CircularAperture(pos, r=r_ap)
        ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
        phot = aperture_photometry(fdata, ap)
        bkg = ApertureStats(fdata, ann, sigma_clip=sclip).median
        net = np.asarray(phot["aperture_sum"]) - np.asarray(bkg) * ap.area

        # frame time
        dobs = fhdr.get("DATE-OBS")
        try:
            mjd = Time(dobs, format="isot").mjd if dobs else np.nan
        except Exception:  # noqa: BLE001
            mjd = np.nan
        times.append(mjd)

        for sid, f_net, b in zip(sids, net, np.asarray(bkg)):
            rows.append((fpath.name, mjd, int(sid), float(f_net), float(b)))

        if (k + 1) % 25 == 0 or k == len(lights) - 1:
            log(f"  ... {k + 1}/{len(lights)} frames "
                f"(median centroid drift this frame: {drift_med[-1]:.2f} px)")

    fm = pd.DataFrame(rows, columns=["frame", "mjd", "source_id", "flux_net", "bkg"])
    fm_path = OUT_DIR / "xval_flux_matrix.csv"
    fm.to_csv(fm_path, index=False)
    log("-" * 72)
    log(f"wrote flux matrix : {fm_path}  ({len(fm)} rows)")

    # ── diagnostics ──────────────────────────────────────────────────────────
    mjds = np.array([t for t in times if np.isfinite(t)])
    span_min = (mjds.max() - mjds.min()) * 24 * 60 if mjds.size > 1 else float("nan")
    pos_frac = float((fm["flux_net"] > 0).mean())
    # quick per-source instrumental scatter (NOT differential — just a sanity peek)
    g_inst = fm.copy()
    g_inst = g_inst[g_inst["flux_net"] > 0]
    g_inst["inst_mag"] = -2.5 * np.log10(g_inst["flux_net"])
    per_src = g_inst.groupby("source_id")["inst_mag"].agg(["count", "std"])
    per_src = per_src[per_src["count"] >= max(5, int(0.5 * len(lights)))]

    log("=" * 72)
    log("DIAGNOSTICS")
    log(f"  frames measured        : {len(lights)}")
    log(f"  time span              : {span_min:.1f} min")
    log(f"  median centroid drift  : {np.nanmedian(drift_med):.2f} px "
        f"(max frame {np.nanmax(drift_med):.2f} px)  <- alignment quality")
    log(f"  positive-flux fraction : {pos_frac * 100:.1f}%")
    log(f"  sources w/ >=50% epochs: {len(per_src)}")
    if len(per_src):
        log(f"  raw instr-mag scatter  : median {per_src['std'].median():.4f} mag "
            f"(NOT yet differential — comp detrending comes in Stage 2)")
    log("=" * 72)

    meta = {
        "generated_utc": t_start.isoformat(),
        "root": str(root), "masterstar": str(master),
        "n_frames": len(lights), "field_ra": cen.ra.deg, "field_dec": cen.dec.deg,
        "radius_deg": radius_deg, "object_header": mhdr.get("OBJECT", None),
        "fwhm_px": fwhm, "r_ap_px": r_ap, "r_in_px": r_in, "r_out_px": r_out,
        "gmax": GMAX, "max_sources": MAX_SOURCES, "n_sources": int(len(gdf)),
        "plate_scale_arcsec_px": 9.768,
        "median_drift_px": float(np.nanmedian(drift_med)),
        "time_span_min": span_min,
    }
    meta_path = OUT_DIR / "xval_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"wrote metadata    : {meta_path}")
    log("DONE — paste this whole output + confirm xval_out/ contents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
