"""REG-520-01 read-only: S1 matching A/B/C, M1b provenance, M2 curve, LC replay.

Writes only under this session directory. Live draft 520 is not modified.
Rig: Brno AZ800 / C5A-150M, non-cal path.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src_py"))

from masterstar_gaia_accounting import (  # noqa: E402
    ForcedSeedAcceptParams,
    Pass2AcceptParams,
    dao_pass2_try_at_position,
    forced_seed_accept,
    forced_seed_measure_at_position,
)

OUT = Path(__file__).resolve().parent
G520 = REPO / "Archive" / "Drafts" / "draft_000520" / "platesolve" / "g_60_4"
MS_FITS = G520 / "MASTERSTAR.fits"
MS_CSV = G520 / "masterstars_full_match.csv"
CENSUS = G520 / "gaia_source_state_census.csv"
COMP_PT = G520 / "photometry" / "comparison_stars_per_target.csv"
PROC = REPO / "Archive" / "Drafts" / "draft_000520" / "detrended_aligned" / "lights" / "g_60_4"
ALIGN_MID = PROC / "SSCam_2026-06-08_21-09-19_g_0048.fits"
RAW_MID = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000520"
    / "non_calibrated"
    / "lights"
    / "g_60_4"
    / "SSCam_2026-06-08_21-09-19_g_0048.fits"
)
EPSF516 = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "masterstar_epsf.fits"
)
TARGET = "1111749368289526912"
EXPECTED_EPSF = "172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20"
FWHM = 1.25
PASS2_SIGMA = 4.0
SAT_80 = 65535.0 * 0.80


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _cid(v: object) -> str:
    s = str(v or "").strip()
    if s.endswith(".0") and s[:-2].replace("-", "").isdigit():
        s = s[:-2]
    return s


def load_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


def lc_from_ensemble(
    frames: list[pd.DataFrame],
    target_cid: str,
    comp_ids: list[str],
    flux_col: str = "dao_flux",
) -> dict:
    times: list[float] = []
    dms: list[float] = []
    n_comp_used: list[int] = []
    want = set(comp_ids)
    for df in frames:
        cid = df["catalog_id"].map(_cid)
        trow = df.loc[cid == target_cid]
        if trow.empty:
            continue
        ft = float(pd.to_numeric(trow.iloc[0][flux_col], errors="coerce"))
        if not (math.isfinite(ft) and ft > 0):
            continue
        crow = df.loc[cid.isin(want)]
        fluxes = pd.to_numeric(crow[flux_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(fluxes) & (fluxes > 0)
        if int(ok.sum()) < 2:
            continue
        fc = float(np.median(fluxes[ok]))
        if not (math.isfinite(fc) and fc > 0):
            continue
        bjd = float(pd.to_numeric(trow.iloc[0].get("bjd_tdb_mid"), errors="coerce"))
        times.append(bjd)
        dms.append(-2.5 * math.log10(ft / fc))
        n_comp_used.append(int(ok.sum()))
    arr = np.asarray(dms, dtype=float)
    if arr.size < 3:
        return {"n": int(arr.size), "lc_rms": None, "lc_rms_ooe": None, "times": times, "dmag": dms}
    med = float(np.median(arr))
    ooe = arr[arr <= np.quantile(arr, 0.33)] if arr.size >= 6 else arr
    return {
        "n": int(arr.size),
        "lc_rms": float(np.std(arr, ddof=0)),
        "lc_rms_ooe": float(np.std(ooe, ddof=0)) if ooe.size >= 3 else None,
        "median_dmag": med,
        "mean_n_comp": float(np.mean(n_comp_used)),
        "times": times,
        "dmag": dms,
    }


def save_lc_plot(path: Path, series: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for name, rec in series.items():
        t = np.asarray(rec.get("times") or [], dtype=float)
        y = np.asarray(rec.get("dmag") or [], dtype=float)
        if t.size == 0:
            continue
        t0 = t - t.min()
        ax.plot(t0, y, "o-", ms=4, label=f"{name} rms={rec.get('lc_rms')}")
    ax.invert_yaxis()
    ax.set_xlabel("BJD - BJD0 (days)")
    ax.set_ylabel("delta mag (T / median comps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_cutout_grid(
    path: Path,
    img: np.ndarray,
    rows: pd.DataFrame,
    *,
    title: str,
    half: int = 18,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(len(rows))
    cols = 4
    rr = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rr, cols, figsize=(cols * 2.4, rr * 2.5))
    axes = np.atleast_2d(axes)
    h, w = img.shape
    for i, (_, row) in enumerate(rows.iterrows()):
        ax = axes[i // cols, i % cols]
        x = float(row["x"])
        y = float(row["y"])
        x0, x1 = max(0, int(round(x)) - half), min(w, int(round(x)) + half + 1)
        y0, y1 = max(0, int(round(y)) - half), min(h, int(round(y)) + half + 1)
        cut = img[y0:y1, x0:x1]
        if cut.size:
            lo, hi = np.percentile(cut[np.isfinite(cut)], [5.0, 99.5])
            ax.imshow(cut, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.axhline(y - y0, color="red", lw=0.6, alpha=0.5)
        ax.axvline(x - x0, color="red", lw=0.6, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        g = row.get("phot_g_mean_mag", row.get("g_mag"))
        ax.set_title(f"{str(row['catalog_id'])[-7:]}\nG={float(g):.2f} {row.get('source_state','')}", fontsize=7)
    for j in range(n, rr * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def match_curve(det_xy: np.ndarray, gaia_xy: np.ndarray, radii: list[float]) -> list[dict]:
    if det_xy.size == 0 or gaia_xy.size == 0:
        return []
    tree = cKDTree(det_xy)
    d, _ = tree.query(gaia_xy, k=1)
    out = []
    n = int(len(d))
    for r in radii:
        n_ok = int(np.sum(d <= float(r)))
        out.append({"radius_px": float(r), "n_gaia": n, "n_matched": n_ok, "frac": n_ok / n if n else None})
    return out


def replay_pass2_seed(data0: np.ndarray, unmatched: pd.DataFrame, tol: float) -> dict:
    p2 = Pass2AcceptParams(sigma=PASS2_SIGMA, center_tol_px=float(tol), fwhm_px=FWHM)
    fs = ForcedSeedAcceptParams(centroid_max_px=float(tol), snr_min=4.0)
    h, w = data0.shape
    n_p2 = 0
    n_seed = 0
    n_rej_tol = 0
    bright_p2 = []
    for _, row in unmatched.iterrows():
        xg = float(row["x_gaia"])
        yg = float(row["y_gaia"])
        gmag = float(row["g_mag"]) if pd.notna(row["g_mag"]) else float("nan")
        r2 = dao_pass2_try_at_position(data0, xg, yg, wpx=w, h=h, params=p2)
        if r2.get("accepted"):
            n_p2 += 1
            if math.isfinite(gmag) and gmag < 14.0:
                bright_p2.append(_cid(row["catalog_id"]))
            continue
        if str(r2.get("reason")) == "centroid_tol":
            n_rej_tol += 1
        meas = forced_seed_measure_at_position(data0, xg, yg, fwhm_px=FWHM, params=fs)
        ok, reason = forced_seed_accept(meas, params=fs)
        if ok:
            n_seed += 1
        elif reason == "centroid_tol":
            n_rej_tol += 1
    return {
        "tol_px": float(tol),
        "n_unmatched_tried": int(len(unmatched)),
        "n_pass2_accept": n_p2,
        "n_forced_seed_accept": n_seed,
        "n_centroid_tol_reject": n_rej_tol,
        "bright_g14_pass2_ids": bright_p2,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gates = {
        "g1_head_parent": "505fa1334fa7be7fa1dc49611c49e29972e0320b",
        "g2_epsf_sha": sha256_file(EPSF516) if EPSF516.is_file() else None,
        "g2_expected": EXPECTED_EPSF,
    }
    gates["g2_match"] = gates["g2_epsf_sha"] == EXPECTED_EPSF
    (OUT / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")

    ms = pd.read_csv(MS_CSV, dtype={"catalog_id": str, "name": str}, low_memory=False)
    ms["catalog_id"] = ms["catalog_id"].map(_cid)
    ms["phot_g_mean_mag"] = pd.to_numeric(ms["phot_g_mean_mag"], errors="coerce")
    ms["peak_dao"] = pd.to_numeric(ms.get("peak_dao"), errors="coerce")
    ms["peak_max_adu"] = pd.to_numeric(ms.get("peak_max_adu"), errors="coerce")
    census = pd.read_csv(CENSUS, dtype={"catalog_id": str})
    census["catalog_id"] = census["catalog_id"].map(_cid)
    census["g_mag"] = pd.to_numeric(census["g_mag"], errors="coerce")
    pt = pd.read_csv(COMP_PT, dtype={"catalog_id": str, "target_catalog_id": str}, low_memory=False)
    pt["catalog_id"] = pt["catalog_id"].map(_cid)
    pt["target_catalog_id"] = pt["target_catalog_id"].map(_cid)
    v0612 = pt[pt["target_catalog_id"] == TARGET].copy()
    selected_ids = [c for c in v0612["catalog_id"].tolist() if c]
    june = ms[(ms["phot_g_mean_mag"] >= 11.6) & (ms["phot_g_mean_mag"] <= 13.9) & (ms["catalog_id"] != TARGET)].copy()
    june_ids = june["catalog_id"].tolist()

    cen_map = census.set_index("catalog_id")
    ms_map = ms.set_index("catalog_id")

    proc_paths = sorted(PROC.glob("proc_*.csv"))
    frames = []
    usecols = [
        "catalog_id",
        "name",
        "bjd_tdb_mid",
        "dao_flux",
        "flux",
        "peak_dao",
        "peak_max_adu",
        "forced_photometry",
        "source_type",
        "source_state",
        "photometry_ok",
        "sigma_bkg_ap",
        "aperture_r_px",
        "snr50_ok",
        "likely_saturated",
        "zone",
        "x",
        "y",
    ]
    for p in proc_paths:
        dfp = pd.read_csv(
            p,
            usecols=lambda c: c in usecols,
            dtype={"catalog_id": str, "name": str},
            low_memory=False,
        )
        dfp["catalog_id"] = dfp["catalog_id"].map(_cid)
        frames.append(dfp)
    proc_all = pd.concat(frames, ignore_index=True)

    def star_forensics(cids: list[str], label: str) -> pd.DataFrame:
        rows = []
        for cid in cids:
            sub = proc_all[proc_all["catalog_id"] == cid]
            msrow = ms_map.loc[cid] if cid in ms_map.index else None
            cenrow = cen_map.loc[cid] if cid in cen_map.index else None
            peak = pd.to_numeric(sub["peak_max_adu"], errors="coerce") if len(sub) else pd.Series(dtype=float)
            flux = pd.to_numeric(sub["dao_flux"], errors="coerce") if len(sub) else pd.Series(dtype=float)
            sig = pd.to_numeric(sub["sigma_bkg_ap"], errors="coerce") if len(sub) else pd.Series(dtype=float)
            ap = pd.to_numeric(sub["aperture_r_px"], errors="coerce") if len(sub) else pd.Series(dtype=float)
            area = math.pi * float(ap.median()) ** 2 if len(ap) and math.isfinite(float(ap.median())) else float("nan")
            snr = flux / (sig * math.sqrt(area)) if math.isfinite(area) and area > 0 else pd.Series(dtype=float)
            forced = sub["forced_photometry"] if "forced_photometry" in sub.columns else pd.Series(dtype=object)
            stype = sub["source_type"].astype(str) if "source_type" in sub.columns else pd.Series(dtype=object)
            rows.append(
                {
                    "set": label,
                    "catalog_id": cid,
                    "phot_g_mean_mag": (
                        float(msrow["phot_g_mean_mag"]) if msrow is not None else None
                    ),
                    "ms_source_state": (
                        str(msrow["source_state"]) if msrow is not None and "source_state" in msrow.index else None
                    ),
                    "census_source_state": (
                        str(cenrow["source_state"]) if cenrow is not None else None
                    ),
                    "ms_source_type": (
                        str(msrow["source_type"]) if msrow is not None else None
                    ),
                    "ms_forced_photometry": (
                        str(msrow["forced_photometry"]) if msrow is not None else None
                    ),
                    "ms_peak_dao": float(msrow["peak_dao"]) if msrow is not None else None,
                    "ms_peak_max_adu": float(msrow["peak_max_adu"]) if msrow is not None else None,
                    "ms_zone": str(msrow["zone"]) if msrow is not None else None,
                    "n_frames": int(len(sub)),
                    "frac_forced_photometry": (
                        float(pd.to_numeric(forced, errors="coerce").fillna(0).mean()) if len(sub) else None
                    ),
                    "source_type_mode": (stype.mode().iloc[0] if len(stype) else None),
                    "per_frame_source_state_persisted": bool("source_state" in sub.columns),
                    "peak_max_median": float(peak.median()) if len(peak) else None,
                    "peak_max_max": float(peak.max()) if len(peak) else None,
                    "sat80_clipped_frac": (
                        float((peak >= SAT_80).mean()) if len(peak) else None
                    ),
                    "snr_ap_median": float(pd.to_numeric(snr, errors="coerce").median()) if len(sub) else None,
                    "x": float(msrow["x"]) if msrow is not None else None,
                    "y": float(msrow["y"]) if msrow is not None else None,
                    "comp_rms_selected_row": (
                        float(v0612.loc[v0612["catalog_id"] == cid, "comp_rms"].iloc[0])
                        if cid in set(v0612["catalog_id"])
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows)

    sel_tab = star_forensics(selected_ids, "selected_today")
    june_tab = star_forensics(june_ids, "june_band_G_11.6_13.9")
    forensic = pd.concat([sel_tab, june_tab], ignore_index=True)
    forensic.to_csv(OUT / "m1b_star_forensics.csv", index=False)

    # Cutouts on MASTERSTAR (same xy as catalog) and one aligned mid light.
    img_ms = load_image(MS_FITS)
    img_al = load_image(ALIGN_MID) if ALIGN_MID.is_file() else img_ms
    sel_xy = sel_tab.dropna(subset=["x", "y"]).copy()
    june_xy = june_tab.dropna(subset=["x", "y"]).sort_values("phot_g_mean_mag").head(8).copy()
    save_cutout_grid(OUT / "cutouts_selected_comps_masterstar.png", img_ms, sel_xy, title="Today selected comps on MASTERSTAR")
    save_cutout_grid(OUT / "cutouts_june_band_masterstar.png", img_ms, june_xy, title="June-band G 11.6-13.9 on MASTERSTAR")
    save_cutout_grid(OUT / "cutouts_selected_comps_aligned_mid.png", img_al, sel_xy, title="Today selected comps on aligned g_0048")
    save_cutout_grid(OUT / "cutouts_june_band_aligned_mid.png", img_al, june_xy, title="June-band on aligned g_0048")

    # M1 matching census on existing catalog + A/B/C pass2/seed replay
    det = ms[ms["source_state"].astype(str).isin(["DETECTED_P1", "DETECTED_P2"])].copy()
    counts_now = {}
    for label, mask in (
        ("Glt12", census["g_mag"] < 12),
        ("Glt14", census["g_mag"] < 14),
        ("G_11.6_13.9", (census["g_mag"] >= 11.6) & (census["g_mag"] <= 13.9) & (census["catalog_id"] != TARGET)),
    ):
        sub = census.loc[mask]
        counts_now[label] = {
            "n_gaia": int(len(sub)),
            "state_counts": sub["source_state"].value_counts().to_dict(),
            "n_detected": int(sub["source_state"].isin(["DETECTED_P1", "DETECTED_P2"]).sum()),
        }

    sky = float(np.nanmedian(img_ms))
    data0 = img_ms - sky
    owned = set(census.loc[census["source_state"].isin(["DETECTED_P1", "DETECTED_P2"]), "catalog_id"])
    unmatched = census.loc[~census["catalog_id"].isin(owned) & census["g_mag"].le(17.5) & ~census["source_state"].eq("EDGE")].copy()

    abc = {}
    for name, tol in (("A_1.0", 1.0), ("B_2.0", 2.0), ("C_2.5", 2.5)):
        abc[name] = replay_pass2_seed(data0, unmatched, tol)
        # n matched G<12 / G<14 = existing P1/P2 plus new pass2 accepts in those bins
        extra = set(abc[name]["bright_g14_pass2_ids"])
        abc[name]["n_detected_Glt12"] = counts_now["Glt12"]["n_detected"]
        abc[name]["n_detected_Glt14"] = counts_now["Glt14"]["n_detected"] + len(extra)
        abc[name]["n_gaia_Glt12"] = counts_now["Glt12"]["n_gaia"]
        abc[name]["n_gaia_Glt14"] = counts_now["Glt14"]["n_gaia"]

    # M1 / M3 LC replay from existing proc photometry (no draft write)
    june_detected = [
        c
        for c in june_ids
        if c in ms_map.index and str(ms_map.loc[c]["source_state"]) in ("DETECTED_P1", "DETECTED_P2")
    ]
    # brightest 8 detected june-band excluding VSX if flagged
    june_det_df = ms.loc[ms["catalog_id"].isin(june_detected)].sort_values("phot_g_mean_mag")
    if "vsx_known_variable" in june_det_df.columns:
        june_det_df = june_det_df[~june_det_df["vsx_known_variable"].astype(str).str.lower().isin(["true", "1"])]
    june_ens = june_det_df["catalog_id"].head(8).tolist()

    lc_today = lc_from_ensemble(frames, TARGET, selected_ids)
    lc_june = lc_from_ensemble(frames, TARGET, june_ens)
    lc_all_june_det = lc_from_ensemble(frames, TARGET, june_detected)
    save_lc_plot(
        OUT / "lc_v0612_ensembles.png",
        {
            "A_today_selected": {k: lc_today[k] for k in lc_today},
            "june_band_detected": {k: lc_june[k] for k in lc_june},
        },
    )

    def _lc_pub(rec: dict) -> dict:
        return {k: v for k, v in rec.items() if k not in ("times", "dmag")}

    m1 = {
        "success_criterion": "B or C restores June-class comps (G 11.6-13.9) and V0612 lc_rms ~0.06",
        "existing_census_bright": counts_now,
        "abc_pass2_seed_replay": abc,
        "selected_ids_today": selected_ids,
        "june_band_ids": june_ids,
        "june_detected_ids": june_detected,
        "june_ensemble_used_for_lc": june_ens,
        "lc_A_today_selected": _lc_pub(lc_today),
        "lc_june_band_detected_8": _lc_pub(lc_june),
        "lc_all_june_detected": _lc_pub(lc_all_june_det),
        "note": (
            "A/B/C vary pass2/seed centroid tol only; match_radius stays 3.5 px. "
            "LC replay uses existing per-frame dao_flux (no re-extraction)."
        ),
    }
    (OUT / "m1_abc.json").write_text(json.dumps(m1, indent=2, default=str), encoding="utf-8")

    # M2 match-fraction vs radius vs solve rms
    gxy = census[["x_gaia", "y_gaia"]].to_numpy(dtype=float)
    dxy = det[["x", "y"]].to_numpy(dtype=float)
    radii = [0.5, 1.0, 1.44, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0]
    curve_all = match_curve(dxy, gxy, radii)
    for band, msk in (
        ("Glt12", census["g_mag"] < 12),
        ("Glt14", census["g_mag"] < 14),
        ("G_11.6_13.9", (census["g_mag"] >= 11.6) & (census["g_mag"] <= 13.9)),
    ):
        sub = census.loc[msk, ["x_gaia", "y_gaia"]].to_numpy(dtype=float)
        curve_all.append({"band": band, "points": match_curve(dxy, sub, radii)})
    # per-set derived vs rms
    sets = {}
    for setup, rms in (("g_60_4", 1.44), ("i_70_4", 2.98), ("r_60_4", 1.49)):
        jp = REPO / "Archive" / "Drafts" / "draft_000520" / "platesolve" / setup / "dao_gaia_calibration.json"
        if not jp.is_file():
            continue
        cert = json.loads(jp.read_text(encoding="utf-8"))
        der = cert.get("derived") or {}
        sets[setup] = {
            "solve_rms_px": rms,
            "fwhm_px": der.get("fwhm_px"),
            "residual_p95_px": der.get("residual_p95_px"),
            "match_radius_px": der.get("match_radius_px"),
            "pass2_center_tol_px": der.get("pass2_center_tol_px"),
            "forced_seed_centroid_max_px": der.get("forced_seed_centroid_max_px"),
            "centroid_floor_px": (cert.get("inputs") or {}).get("centroid_floor_px"),
            "tol_minus_rms": float(der.get("pass2_center_tol_px") or 0) - float(rms),
        }
    m2 = {
        "principle": "derived matching/centroid tolerance must be a function of measured astrometric rms and FWHM; never a fixed 1.0 px floor",
        "g_match_curve_all_gaia": [x for x in curve_all if "radius_px" in x],
        "g_match_curve_by_band": [x for x in curve_all if "band" in x],
        "per_set_derived_vs_solve_rms": sets,
        "floor_code": "dao_gaia_calibration.py:36 DEFAULT_CENTROID_FLOOR_PX=1.0; :728-763 derive_tolerances_from_diagnostic",
    }
    (OUT / "m2_tolerance_curve.json").write_text(json.dumps(m2, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = [p["radius_px"] for p in m2["g_match_curve_all_gaia"]]
    ys = [p["frac"] for p in m2["g_match_curve_all_gaia"]]
    ax.plot(xs, ys, "k-o", label="all on-chip Gaia")
    for band_rec in m2["g_match_curve_by_band"]:
        pts = band_rec["points"]
        ax.plot([p["radius_px"] for p in pts], [p["frac"] for p in pts], "o-", label=band_rec["band"])
    ax.axvline(1.0, color="red", ls="--", label="derived centroid floor 1.0")
    ax.axvline(1.44, color="orange", ls=":", label="g solve rms 1.44")
    ax.axvline(2.0, color="green", ls="--", label="hand-class 2.0")
    ax.set_xlabel("match radius (px)")
    ax.set_ylabel("fraction of Gaia with a DETECTED neighbour")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "m2_match_vs_radius.png", dpi=120)
    plt.close(fig)

    sel_forced = float(sel_tab["frac_forced_photometry"].mean()) if len(sel_tab) else None
    sel_det = (
        int((sel_tab["census_source_state"].isin(["DETECTED_P1", "DETECTED_P2"])).sum())
        if len(sel_tab)
        else 0
    )
    m1b = {
        "n_selected": int(len(sel_tab)),
        "n_selected_detected_p1p2": sel_det,
        "frac_selected_detected": sel_det / len(sel_tab) if len(sel_tab) else None,
        "mean_frac_forced_photometry_selected": sel_forced,
        "per_frame_source_state_column_present": bool(
            sel_tab["per_frame_source_state_persisted"].all()
        )
        if len(sel_tab)
        else False,
        "comp_pool_gates_on_source_state": False,
        "comp_pool_gate_file_line": "photometry_core.py:15777-15860 cand_mask has no source_state; admit_pool_stars :1002-1046 VSX/Gaia-variable only; _select_comps_by_rms_then_color :15392 uses phase01_comparison_max_comp_rms=0.1 ceiling",
        "max_comp_rms_config": 0.1,
        "june_band_n": int(len(june_tab)),
        "june_band_detected_n": int(
            (june_tab["census_source_state"].isin(["DETECTED_P1", "DETECTED_P2"])).sum()
        ),
        "june_band_peak_max_median": float(june_tab["peak_max_median"].median()) if len(june_tab) else None,
        "june_band_sat80_any": float(june_tab["sat80_clipped_frac"].max()) if len(june_tab) else None,
        "selected_peak_max_median": float(sel_tab["peak_max_median"].median()) if len(sel_tab) else None,
    }
    (OUT / "m1b_summary.json").write_text(json.dumps(m1b, indent=2, default=str), encoding="utf-8")

    print("gates", gates)
    print("selected states", sel_tab["census_source_state"].value_counts().to_dict())
    print("june states", june_tab["census_source_state"].value_counts().to_dict())
    print("abc", {k: {kk: vv for kk, vv in v.items() if kk != "bright_g14_pass2_ids"} for k, v in abc.items()})
    print("lc today", _lc_pub(lc_today))
    print("lc june8", _lc_pub(lc_june))
    print("m1b", m1b)


if __name__ == "__main__":
    main()
