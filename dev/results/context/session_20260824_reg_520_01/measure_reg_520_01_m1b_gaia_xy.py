"""REG-520-01: Gaia-xy residuals, cutouts, G<14 complete IDs, forensics CSV."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src_py"))

G520 = REPO / "Archive" / "Drafts" / "draft_000520" / "platesolve" / "g_60_4"
PROC = REPO / "Archive" / "Drafts" / "draft_000520" / "detrended_aligned" / "lights" / "g_60_4"
LIGHTS = REPO / "Archive" / "Drafts" / "draft_000520" / "non_calibrated" / "lights" / "g_60_4"
ALIGNED = REPO / "Archive" / "Drafts" / "draft_000520" / "detrended_aligned" / "lights" / "g_60_4"
TARGET = "1111749368289526912"
HALF = 22
SELECTED = [
    "1112112413285008896",
    "1112115024625070720",
    "1111930718988511616",
    "1112119250872867200",
    "1112110042463052928",
    "1111931371821079552",
    "1111737823417422464",
    "1111922300852743808",
]


def cid(v: object) -> str:
    s = str(v or "").strip()
    if s.endswith(".0") and s[:-2].replace("-", "").isdigit():
        s = s[:-2]
    return s


def lc_from(frames, target, comps, flux_col="dao_flux"):
    dms = []
    nused = []
    want = set(comps)
    for df in frames:
        c = df["catalog_id"].map(cid)
        trow = df.loc[c == target]
        if trow.empty:
            continue
        ft = float(pd.to_numeric(trow.iloc[0][flux_col], errors="coerce"))
        if not (math.isfinite(ft) and ft > 0):
            continue
        fl = pd.to_numeric(df.loc[c.isin(want), flux_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(fl) & (fl > 0)
        if int(ok.sum()) < 2:
            continue
        fc = float(np.median(fl[ok]))
        dms.append(-2.5 * math.log10(ft / fc))
        nused.append(int(ok.sum()))
    arr = np.asarray(dms, dtype=float)
    ooe = arr[arr <= np.quantile(arr, 0.33)] if arr.size >= 6 else arr
    return {
        "n": int(arr.size),
        "lc_rms": float(np.std(arr)) if arr.size else None,
        "lc_rms_ooe": float(np.std(ooe)) if ooe.size >= 3 else None,
        "mean_n_comp": float(np.mean(nused)) if nused else None,
        "n_comp_ids": len(comps),
        "dmag": [float(x) for x in arr],
    }


def cutout_grid(img: np.ndarray, rows: pd.DataFrame, xcol: str, ycol: str, title: str, out: Path) -> None:
    h, w = img.shape
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.2))
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        x, y = float(row[xcol]), float(row[ycol])
        x0, x1 = max(0, int(round(x)) - HALF), min(w, int(round(x)) + HALF + 1)
        y0, y1 = max(0, int(round(y)) - HALF), min(h, int(round(y)) + HALF + 1)
        cut = img[y0:y1, x0:x1]
        finite = cut[np.isfinite(cut)]
        lo, hi = (np.percentile(finite, [5, 99.5]) if finite.size else (0.0, 1.0))
        ax.imshow(cut, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.axhline(y - y0, color="lime", lw=0.6)
        ax.axvline(x - x0, color="lime", lw=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        gid = str(row["catalog_id"])[-7:]
        gmag = float(row["g"])
        d = float(row["d_gaia_px"])
        ax.set_title(f"{gid} G={gmag:.2f}\nDAO-Gaia={d:.0f}px", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close()


def main() -> None:
    abc = json.loads((HERE / "m1_abc.json").read_text(encoding="utf-8"))
    ms = pd.read_csv(G520 / "masterstars_full_match.csv", dtype={"catalog_id": str})
    ce = pd.read_csv(G520 / "gaia_source_state_census.csv", dtype={"catalog_id": str})
    ms["catalog_id"] = ms["catalog_id"].map(cid)
    ce["catalog_id"] = ce["catalog_id"].map(cid)
    m = ms.merge(ce, on="catalog_id", suffixes=("", "_cen"))
    m["dx"] = pd.to_numeric(m["x"], errors="coerce") - pd.to_numeric(m["x_gaia"], errors="coerce")
    m["dy"] = pd.to_numeric(m["y"], errors="coerce") - pd.to_numeric(m["y_gaia"], errors="coerce")
    m["d_gaia_px"] = np.hypot(m["dx"], m["dy"])
    m["g"] = pd.to_numeric(m["phot_g_mean_mag"], errors="coerce")

    frames = []
    proc_xy_rows = []
    for p in sorted(PROC.glob("proc_*.csv")):
        df = pd.read_csv(
            p,
            usecols=lambda c: c
            in {
                "catalog_id",
                "dao_flux",
                "flux",
                "x",
                "y",
                "forced_photometry",
                "source_type",
                "peak_dao",
                "snr_aperture_mode",
                "bjd_tdb_mid",
            },
            dtype={"catalog_id": str},
        )
        df["catalog_id"] = df["catalog_id"].map(cid)
        frames.append(df)
        sub = df.loc[df["catalog_id"].isin(SELECTED)].copy()
        sub["proc_file"] = p.name
        proc_xy_rows.append(sub)
    nfr = len(frames)
    counts: dict[str, int] = {}
    for df in frames:
        for c, fl in zip(df["catalog_id"], pd.to_numeric(df["dao_flux"], errors="coerce")):
            if math.isfinite(float(fl)) and float(fl) > 0:
                counts[c] = counts.get(c, 0) + 1

    g14_complete = [
        c
        for c in m.loc[
            (m["g"] < 14)
            & (m["catalog_id"] != TARGET)
            & (m["source_state"].isin(["DETECTED_P1", "DETECTED_P2"])),
            "catalog_id",
        ]
        if counts.get(c, 0) >= 20
    ]
    g14_complete8 = sorted(
        g14_complete,
        key=lambda c: float(m.loc[m["catalog_id"] == c, "g"].iloc[0]),
    )[:8]
    g14_rows = m.loc[m["catalog_id"].isin(g14_complete8), ["catalog_id", "g", "d_gaia_px", "x", "y", "x_gaia", "y_gaia"]].copy()
    g14_rows = g14_rows.sort_values("g")

    sel = m.loc[m["catalog_id"].isin(SELECTED)].copy()
    sel = sel.set_index("catalog_id").loc[SELECTED].reset_index()

    img = np.asarray(fits.getdata(G520 / "MASTERSTAR.fits"), dtype=np.float64)
    cutout_grid(
        img,
        sel,
        "x_gaia",
        "y_gaia",
        "Selected comps at CENSUS Gaia xy (green cross); MASTERSTAR",
        HERE / "cutouts_selected_at_gaia_xy.png",
    )
    mid_fits = sorted(LIGHTS.glob("SSCam_*.fits")) or sorted(ALIGNED.glob("SSCam_*.fits"))
    mid = mid_fits[len(mid_fits) // 2] if mid_fits else None
    if mid is not None:
        img_mid = np.asarray(fits.getdata(mid), dtype=np.float64)
        cutout_grid(
            img_mid,
            sel,
            "x_gaia",
            "y_gaia",
            f"Selected comps at CENSUS Gaia xy; aligned mid {mid.name}",
            HERE / "cutouts_selected_at_gaia_xy_aligned_mid.png",
        )

    proc_all = pd.concat(proc_xy_rows, ignore_index=True) if proc_xy_rows else pd.DataFrame()
    proc_xy_summary = []
    if not proc_all.empty:
        for cid_s in SELECTED:
            sub = proc_all.loc[proc_all["catalog_id"] == cid_s]
            ms_row = sel.loc[sel["catalog_id"] == cid_s].iloc[0]
            px = pd.to_numeric(sub["x"], errors="coerce")
            py = pd.to_numeric(sub["y"], errors="coerce")
            d_ms = np.hypot(px - float(ms_row["x"]), py - float(ms_row["y"]))
            d_ga = np.hypot(px - float(ms_row["x_gaia"]), py - float(ms_row["y_gaia"]))
            fp = pd.to_numeric(sub["forced_photometry"], errors="coerce")
            proc_xy_summary.append(
                {
                    "catalog_id": cid_s,
                    "n_proc": int(len(sub)),
                    "forced_frac": float(fp.fillna(0).mean()) if len(sub) else None,
                    "proc_xy_vs_ms_median_px": float(np.nanmedian(d_ms)) if len(sub) else None,
                    "proc_xy_vs_gaia_median_px": float(np.nanmedian(d_ga)) if len(sub) else None,
                    "source_type_mode": str(sub["source_type"].mode().iloc[0]) if len(sub) and sub["source_type"].notna().any() else None,
                }
            )
    (HERE / "m1b_selected_proc_xy.json").write_text(json.dumps(proc_xy_summary, indent=2), encoding="utf-8")

    cal = pd.read_csv(
        REPO / "dev" / "results" / "session_20260824_cal_520_01" / "m4_comp_forensics.csv",
        dtype={"catalog_id": str},
    )
    cal["catalog_id"] = cal["catalog_id"].map(cid)
    cal_map = cal.set_index("catalog_id")

    forensic_rows = []
    for label, ids in (
        ("selected_today", SELECTED),
        ("june_band_G_11.6_13.9", abc["june_band_ids"]),
        ("g14_complete8", g14_complete8),
    ):
        for c in ids:
            hit = m.loc[m["catalog_id"] == c]
            if hit.empty:
                continue
            row = hit.iloc[0]
            cal_r = cal_map.loc[c] if c in cal_map.index else None
            forensic_rows.append(
                {
                    "set": label,
                    "catalog_id": c,
                    "phot_g_mean_mag": float(row["g"]) if math.isfinite(float(row["g"])) else None,
                    "source_state": str(row.get("source_state", "")),
                    "ms_x": float(row["x"]) if math.isfinite(float(row["x"])) else None,
                    "ms_y": float(row["y"]) if math.isfinite(float(row["y"])) else None,
                    "x_gaia": float(row["x_gaia"]) if math.isfinite(float(row["x_gaia"])) else None,
                    "y_gaia": float(row["y_gaia"]) if math.isfinite(float(row["y_gaia"])) else None,
                    "d_gaia_px": float(row["d_gaia_px"]) if math.isfinite(float(row["d_gaia_px"])) else None,
                    "n_frames_dao_flux": int(counts.get(c, 0)),
                    "comp_rms_fieldwide": (
                        float(cal_r["comp_rms_fieldwide_today"])
                        if cal_r is not None and "comp_rms_fieldwide_today" in cal_r.index and pd.notna(cal_r["comp_rms_fieldwide_today"])
                        else None
                    ),
                    "exclusion_reason": (
                        str(cal_r["exclusion_reason"])
                        if cal_r is not None and "exclusion_reason" in cal_r.index and pd.notna(cal_r["exclusion_reason"])
                        else None
                    ),
                    "selected_for_V0612": c in set(SELECTED),
                }
            )
    forensic = pd.DataFrame(forensic_rows)
    forensic.to_csv(HERE / "june_comp_forensics.csv", index=False)
    sel[["catalog_id", "g", "x", "y", "x_gaia", "y_gaia", "d_gaia_px", "source_state"]].to_csv(
        HERE / "m1b_selected_xy_vs_gaia.csv", index=False
    )

    lc_g14 = lc_from(frames, TARGET, g14_complete8)
    lc_today = lc_from(frames, TARGET, SELECTED)
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(lc_today["dmag"], "o", ms=4, label=f"today selected 8  lc_rms={lc_today['lc_rms']:.3f}")
    ax.plot(lc_g14["dmag"], "s", ms=4, label=f"G<14 complete 8  lc_rms={lc_g14['lc_rms']:.3f}")
    ax.set_xlabel("epoch index (g_60_4, 24 frames with target flux)")
    ax.set_ylabel("dmag (target - median ensemble)")
    ax.set_title("V0612 Cam g_60_4: today selected vs G<14 complete (same dao_flux)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "lc_v0612_today_vs_g14_complete8.png", dpi=130)
    plt.close()

    g14_meta = [
        {
            "catalog_id": c,
            "g": float(m.loc[m["catalog_id"] == c, "g"].iloc[0]),
            "d_gaia_px": float(m.loc[m["catalog_id"] == c, "d_gaia_px"].iloc[0]),
            "n_frames": int(counts.get(c, 0)),
        }
        for c in g14_complete8
    ]
    m3 = json.loads((HERE / "m3_bright_ensemble.json").read_text(encoding="utf-8"))
    m3["g14_complete8"] = g14_complete8
    m3["g14_complete8_meta"] = g14_meta
    m3["lc_g14_complete8"] = {k: v for k, v in lc_g14.items() if k != "dmag"}
    m3["lc_today_selected"] = {k: v for k, v in lc_today.items() if k != "dmag"}
    (HERE / "m3_bright_ensemble.json").write_text(json.dumps(m3, indent=2), encoding="utf-8")

    abc_table = pd.DataFrame(
        [
            {
                "variant": k,
                **{kk: vv for kk, vv in v.items() if kk != "bright_g14_pass2_ids"},
            }
            for k, v in abc["abc_pass2_seed_replay"].items()
        ]
    )
    abc_table.to_csv(HERE / "m1_abc_table.csv", index=False)

    out = {
        "n_frames": nfr,
        "g14_complete8": g14_complete8,
        "g14_complete8_meta": g14_meta,
        "selected_d_gaia_median": float(sel["d_gaia_px"].median()),
        "selected_n_d_le_2": int((sel["d_gaia_px"] <= 2).sum()),
        "proc_xy_summary": proc_xy_summary,
        "lc_g14_complete8": {k: v for k, v in lc_g14.items() if k != "dmag"},
        "lc_today_selected": {k: v for k, v in lc_today.items() if k != "dmag"},
    }
    (HERE / "m1b_gaia_xy_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "proc_xy_summary"}, indent=2))
    print("proc_xy_summary", json.dumps(proc_xy_summary, indent=2))


if __name__ == "__main__":
    main()
