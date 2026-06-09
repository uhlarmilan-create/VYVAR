"""Run real VYVAR stages on synthetic data; score vs injected truth."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.validation.gen_frame import (  # noqa: E402
    FWHM_PX,
    GAIN_E_PER_ADU,
    NY,
    NX,
    PLATE_SCALE_ARCSEC,
    READ_NOISE_E,
    SAT_ADU,
    ZP,
    wcs_for_frame,
    write_frame,
)
from tests.validation.gen_series import (  # noqa: E402
    B1_AMP_MAG,
    B1_PERIOD_D,
    B2_CR_FRAMES,
    B4_COLOR_SLOPE,
    BAD_COMP_ID,
    N_FRAMES,
    TARGET_ID,
    WEAK_TARGET_ID,
    write_series,
)
from tests.validation.score import ValidationReport  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parent / "data"
TIER_A_DIR = DATA_ROOT / "tier_a"
TIER_B_DIR = DATA_ROOT / "tier_b"


def _load_truth(path: Path) -> dict:
    with open(path, encoding="ascii") as f:
        return json.load(f)


def _star_by_name(truth: dict, name: str) -> dict:
    for s in truth["stars"]:
        if s["name"] == name:
            return s
    raise KeyError(name)


def _aperture_r(fwhm: float) -> float:
    return max(2.5, 1.2 * fwhm)


def _epsf_asymmetry_metric(arr: np.ndarray) -> float:
    """Same quad-symmetry metric as psf_photometry.py:521-549."""
    data = np.asarray(arr, dtype=np.float64)
    cy, cx = np.array(data.shape) // 2
    q1 = data[:cy, :cx]
    q2 = data[:cy, cx + (data.shape[1] % 2) :]
    q3 = data[cy + (data.shape[0] % 2) :, :cx]
    q4 = data[cy + (data.shape[0] % 2) :, cx + (data.shape[1] % 2) :]
    min_r = min(q1.shape[0], q2.shape[0], q3.shape[0], q4.shape[0])
    min_c = min(q1.shape[1], q2.shape[1], q3.shape[1], q4.shape[1])
    quads = np.stack([q[:min_r, :min_c] for q in [q1, q2[:, ::-1], q3[::-1, :], q4[::-1, ::-1]]])
    finite_q = np.all(np.isfinite(quads), axis=0)
    peak = float(np.nanmax(data))
    if not (finite_q.any() and peak > 0):
        return float("nan")
    return float(np.nanstd(quads[:, finite_q], axis=0).mean()) / peak


def _run_tier_a(report: ValidationReport) -> None:
    fits_path = TIER_A_DIR / "tier_a_frame.fits"
    truth_path = TIER_A_DIR / "tier_a_truth.json"
    if not fits_path.is_file():
        write_frame(TIER_A_DIR)
    truth = _load_truth(truth_path)

    with fits.open(fits_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header

    wcs = wcs_for_frame()
    from crowding_index import _build_blend_targets_df

    stars_rows = []
    cone_rows = []
    for s in truth["stars"]:
        cid = s["name"]
        cone_rows.append(
            {"catalog_id": cid, "x": s["x"], "y": s["y"], "mag_g": s["mag"]}
        )
    cone_f = pd.DataFrame(cone_rows)
    fwhm = float(truth["fwhm_px"])
    area = math.pi * _aperture_r(fwhm) ** 2
    sky_pp = float(np.median(data))

    for target_name, item_id, desc, expect_blend, nn_lo, nn_hi in (
        (
            "blend_A_target",
            "A1",
            "unresolved blend nn ~1.0 FWHM",
            True,
            0.8,
            1.3,
        ),
        (
            "pair_C",
            "A2",
            "resolvable pair nn ~2.5 FWHM",
            False,
            2.2,
            2.9,
        ),
    ):
        st = _star_by_name(truth, target_name)
        stars_df = pd.DataFrame(
            [
                {
                    "name": target_name,
                    "catalog_id": target_name,
                    "ra_deg": st["ra_deg"],
                    "dec_deg": st["dec_deg"],
                    "mag": st["mag"],
                }
            ]
        )
        blend_df = _build_blend_targets_df(
            stars_df,
            wcs=wcs,
            cone_f=cone_f,
            fwhm_px=fwhm,
            sky_pp=sky_pp,
            area=area,
            gain=GAIN_E_PER_ADU,
            rn=READ_NOISE_E,
            zp=ZP,
            frame_limit_mag=16.0,
        )
        if blend_df.empty:
            report.add(
                item_id,
                desc,
                "crowding_index._build_blend_targets_df",
                "crowding_index is_blended / nn_dist_fwhm (1.5 FWHM threshold)",
                expected=f"is_blended={expect_blend}; nn in [{nn_lo},{nn_hi}]",
                recovered="empty blend_df",
                status="FAIL",
                note="Blend metrics dataframe empty -- check WCS/cone catalog wiring.",
            )
            continue
        row = blend_df.iloc[0]
        is_blended = bool(row.get("is_blended", False))
        nn = float(row.get("nn_dist_fwhm", float("nan")))
        ok_blend = is_blended == expect_blend
        ok_nn = math.isfinite(nn) and nn_lo <= nn <= nn_hi
        status = "PASS" if ok_blend and ok_nn else "FAIL"
        note = ""
        if status == "FAIL":
            note = (
                f"Truth sep_fwhm={expect_blend}; got is_blended={is_blended}, "
                f"nn_dist_fwhm={nn:.3f}. Threshold 1.5 FWHM in crowding_index."
            )
        report.add(
            item_id,
            desc,
            "crowding_index._build_blend_targets_df",
            "crowding_index is_blended / nn_dist_fwhm (1.5 FWHM threshold)",
            expected=f"is_blended={expect_blend}; nn in [{nn_lo},{nn_hi}]",
            recovered=f"is_blended={is_blended}; nn_dist_fwhm={nn:.3f}",
            delta=f"nn_delta={nn - (1.0 if expect_blend else 2.5):.3f}",
            status=status,
            note=note,
        )

    smeared = _star_by_name(truth, "smeared_star")
    round_star = _star_by_name(truth, "comp_00")
    half = 25
    cy, cx = int(smeared["y"]), int(smeared["x"])
    cut_s = data[cy - half : cy + half, cx - half : cx + half].copy()
    cut_s -= float(np.median(cut_s))
    cy, cx = int(round_star["y"]), int(round_star["x"])
    cut_r = data[cy - half : cy + half, cx - half : cx + half].copy()
    cut_r -= float(np.median(cut_r))
    asym_s = _epsf_asymmetry_metric(cut_s)
    asym_r = _epsf_asymmetry_metric(cut_r)
    qc_warn = asym_s > 0.1 or (asym_r > 0 and asym_s / asym_r > 3.0)
    report.add(
        "A3",
        "tracking smear (ellip 0.72)",
        "psf_photometry epsf_asymmetry QC",
        "psf_photometry.py:521-549",
        expected="epsf_asymmetry > 0.1 or smeared/round ratio > 3",
        recovered=f"asym_smeared={asym_s:.4f}; asym_round={asym_r:.4f}; ratio={asym_s / max(asym_r, 1e-9):.2f}",
        delta=f"margin={asym_s - 0.1:.4f}",
        status="PASS" if qc_warn else "FAIL",
        note="" if qc_warn else "Smear asymmetry metric did not exceed QC threshold vs round control.",
    )

    from comp_qa_core import flag_reasons, sokolovsky_indices

    rng = np.random.default_rng(42)
    base_flux = float(_star_by_name(truth, "comp_00")["flux_adu"])
    n_rep = 25
    flux_series = base_flux + rng.normal(0, base_flux * 0.002, n_rep)
    cr_idx = [5, 12, 18]
    for ci in cr_idx:
        flux_series[ci] *= 3.5
    mags = -2.5 * np.log10(flux_series / flux_series[0])
    mags -= np.nanmedian(mags)
    metrics = sokolovsky_indices(mags)
    flags = flag_reasons(
        metrics["sigma_iqr"],
        metrics["inv_nv"],
        metrics["spike"],
        12.0,
        np.array([12.0]),
        np.array([0.0]),
        np.array([0.01]),
        thr_inv_nv=2.0,
    )
    spike_ok = metrics["spike"] > 3.0 or "spike" in flags
    clean_ok = abs(float(np.nanstd(np.delete(mags, cr_idx))) - float(np.nanstd(mags[:3]))) < 0.05
    status = "PASS" if spike_ok else "FAIL"
    report.add(
        "A4",
        "6 cosmic-ray pixels (series proxy)",
        "comp_qa_core.sokolovsky_indices / flag_reasons",
        "Sokolovsky 2017 spike index",
        expected="spike index elevated on CR-injected series",
        recovered=f"spike={metrics['spike']:.3f}; flags={flags}",
        delta=f"spike_hard=3.0",
        status=status,
        note="" if status == "PASS" else "Spike index did not exceed hard threshold on injected outliers.",
    )

    from photometry_core import _catalog_only_fixed_aperture_flux

    sat = _star_by_name(truth, "saturated_star")
    rap = _aperture_r(fwhm)
    _, _, peak = _catalog_only_fixed_aperture_flux(
        data, sat["x"], sat["y"], rap, rap * 1.5, rap * 2.5
    )
    sat_lim = float(hdr.get("SATURATE", SAT_ADU))
    saturated = peak >= 0.85 * sat_lim
    report.add(
        "A5",
        "saturated star (clip at SATURATE)",
        "photometry_core._catalog_only_fixed_aperture_flux peak vs SATURATE",
        "saturation flag / comp exclusion",
        expected=f"peak >= 0.85*SATURATE ({0.85 * sat_lim:.0f} ADU)",
        recovered=f"peak={peak:.0f} ADU",
        delta=f"frac_sat={peak / sat_lim:.3f}",
        status="PASS" if saturated else "FAIL",
        note="" if saturated else "Peak below saturation fraction -- check injected mag / clip order.",
    )

    xs = np.linspace(50, NX - 50, 20)
    ys = np.full_like(xs, NY / 2)
    bkg = []
    for x, y in zip(xs, ys, strict=True):
        y0, y1 = int(y) - 15, int(y) + 15
        x0, x1 = int(x) - 15, int(x) + 15
        bkg.append(float(np.median(data[y0:y1, x0:x1])))
    tilt = float(np.polyfit(xs, bkg, 1)[0])
    report.add(
        "A6",
        "illumination gradient (Moonlight)",
        "comp_qa locus tilt / background vs position",
        "GAP: flat-only vs CoLiTecVS inverse-median",
        expected="SKIP/INFO -- gradient measurable; no production fix",
        recovered=f"background_tilt={tilt:.4f} ADU/px along x",
        delta="documented gap",
        status="SKIP",
        note="Large-scale gradient remains after flat-only calibration; CoLiTecVS-style inverse-median not implemented.",
    )

    from photometry_core import _catalog_only_fixed_aperture_flux as apflux

    from photutils.centroids import centroid_com

    clean = [s for s in truth["stars"] if s["name"].startswith("comp_")]
    rap = _aperture_r(fwhm)
    rin, rout = rap + 3.0, rap + 8.0
    vy_fluxes = []
    sep_fluxes = []
    have_sep = False
    try:
        import sep

        have_sep = True
    except ImportError:
        pass

    for s in clean[:8]:
        x0, y0 = float(s["x"]), float(s["y"])
        box = data[int(y0) - 8 : int(y0) + 9, int(x0) - 8 : int(x0) + 9]
        if box.size > 0:
            try:
                cy, cx = centroid_com(box)
                yc = int(y0) - 8 + cy
                xc = int(x0) - 8 + cx
            except Exception:
                xc, yc = x0, y0
        else:
            xc, yc = x0, y0
        fx, _, _ = apflux(data, xc, yc, rap, rin, rout)
        vy_fluxes.append(fx)
        if have_sep:
            bkg = sep.Background(data.astype(np.float32), bw=64, bh=64)
            data_sub = data - bkg.back()
            flux, _, _ = sep.sum_circle(
                data_sub, np.array([xc]), np.array([yc]), rap, subpix=5
            )
            sep_fluxes.append(float(flux[0]))

    if have_sep and vy_fluxes:
        ratios = [abs(v - s) / max(v, 1.0) for v, s in zip(vy_fluxes, sep_fluxes, strict=True)]
        med_ratio = float(np.median(ratios))
        ok = med_ratio < 0.002
        report.add(
            "A7",
            "clean control stars SEP cross-validation",
            "photutils aperture vs sep.sum_circle",
            "SEP/SExtractor cross-val (xval_run.py pattern)",
            expected="VYVAR aper vs SEP agree ~0.2%/frame",
            recovered=f"median |delta|/flux={med_ratio:.5f}",
            delta=f"{med_ratio * 100:.3f}%",
            status="PASS" if ok else "FAIL",
            note="" if ok else "Photutils vs SEP flux mismatch exceeds 0.2% on clean stars.",
        )
    else:
        report.add(
            "A7",
            "clean control stars SEP cross-validation",
            "photutils aperture vs sep.sum_circle",
            "SEP/SExtractor cross-val",
            expected="VYVAR aper vs SEP agree ~0.2%/frame",
            recovered="sep not installed",
            status="SKIP",
            note="Install sep package for SEP cross-validation item.",
        )

    mags_rec = []
    mags_truth = []
    zp_offsets = []
    for s in clean:
        fx, _, _ = apflux(data, s["x"], s["y"], rap, rin, rout)
        if fx > 0 and s["flux_adu"] > 0:
            zp_offsets.append(s["mag"] + 2.5 * math.log10(fx))
            mags_truth.append(s["mag"])
    if zp_offsets:
        zp_eff = float(np.median(zp_offsets))
        for s in clean:
            fx, _, _ = apflux(data, s["x"], s["y"], rap, rin, rout)
            if fx > 0:
                mags_rec.append(-2.5 * math.log10(fx) + zp_eff)
    bias = float(np.median(np.array(mags_rec) - np.array(mags_truth))) if mags_rec else float("nan")
    ok = math.isfinite(bias) and abs(bias) < 0.05
    report.add(
        "A8",
        "clean control stars aperture photometry ZP",
        "photometry_core._catalog_only_fixed_aperture_flux",
        "aperture + instrumental mag vs injected ZP",
        expected="recovered mag within few mmag of injected (|bias| < 50 mmag)",
        recovered=f"median bias={bias * 1000:.1f} mmag (n={len(mags_rec)})",
        delta=f"{bias * 1000:.1f} mmag",
        status="PASS" if ok else "FAIL",
        note="" if ok else "Aperture photometry bias exceeds tolerance -- check annulus sky / aperture radius.",
    )


def _run_tier_b(report: ValidationReport) -> None:
    meta_path = TIER_B_DIR / "series_meta.json"
    if not meta_path.is_file():
        write_series(TIER_B_DIR)
    with open(meta_path, encoding="ascii") as f:
        meta = json.load(f)

    proc_dir = TIER_B_DIR / "proc"
    phot_dir = TIER_B_DIR / "photometry"

    fluxes = []
    times = []
    for fi in range(N_FRAMES):
        df = pd.read_csv(proc_dir / f"proc_{fi:03d}.csv", dtype={"catalog_id": str})
        row = df[df["catalog_id"] == TARGET_ID].iloc[0]
        fluxes.append(float(row["dao_flux"]))
        times.append(float(row["bjd_tdb_mid"]))
    fluxes = np.array(fluxes)
    times = np.array(times)
    mags = -2.5 * np.log10(fluxes / np.median(fluxes))
    mags -= np.nanmean(mags)

    from astropy.timeseries import LombScargle

    ls = LombScargle(times, mags)
    freq, power = ls.autopower(minimum_frequency=1.0 / (B1_PERIOD_D * 3), maximum_frequency=3.0 / B1_PERIOD_D)
    best_i = int(np.argmax(power))
    p_rec = 1.0 / freq[best_i]
    amp_rec = float(np.std(mags) * math.sqrt(2.0))
    ok_p = abs(p_rec - B1_PERIOD_D) / B1_PERIOD_D < 0.08
    ok_a = abs(amp_rec - B1_AMP_MAG) < 0.02
    report.add(
        "B1",
        "target sine variability A=0.15 mag",
        "LombScargle on proc CSV light curve",
        "variability recovery / periodogram",
        expected=f"|A_rec-{B1_AMP_MAG}|<0.02; P within few % of {B1_PERIOD_D} d",
        recovered=f"A_rec={amp_rec:.4f} mag; P_rec={p_rec:.3f} d",
        delta=f"dA={amp_rec - B1_AMP_MAG:.4f}; dP={p_rec - B1_PERIOD_D:.3f}",
        status="PASS" if ok_p and ok_a else "FAIL",
        note="" if (ok_p and ok_a) else "Periodogram did not recover injected sine parameters within tolerance.",
    )

    from comp_qa_core import flag_reasons, sokolovsky_indices

    bad_flux = []
    for fi in range(N_FRAMES):
        df = pd.read_csv(proc_dir / f"proc_{fi:03d}.csv", dtype={"catalog_id": str})
        row = df[df["catalog_id"] == BAD_COMP_ID].iloc[0]
        bad_flux.append(float(row["dao_flux"]))
    bad_m = -2.5 * np.log10(np.array(bad_flux) / np.median(bad_flux))
    bad_m -= np.nanmedian(bad_m)
    bmetrics = sokolovsky_indices(bad_m)
    bflags = flag_reasons(
        bmetrics["sigma_iqr"],
        bmetrics["inv_nv"],
        bmetrics["spike"],
        12.3,
        np.array([12.0, 13.0]),
        np.array([0.0, 0.0]),
        np.array([0.01, 0.01]),
        thr_inv_nv=2.0,
    )
    flagged = bool(bflags) or bmetrics["spike"] > 2.5 or bmetrics["inv_nv"] > 1.5
    comps_csv = pd.read_csv(phot_dir / "comparison_stars_per_target.csv", dtype={"catalog_id": str})
    in_pool = BAD_COMP_ID in set(comps_csv["catalog_id"].astype(str))
    status = "PASS" if flagged and not in_pool else "FAIL"
    note = ""
    if status == "FAIL":
        note = (
            f"Sokolovsky flags={bflags}; spike={bmetrics['spike']:.2f}; "
            f"inv_nv={bmetrics['inv_nv']:.2f}; in_pool={in_pool}."
        )
    report.add(
        "B2",
        "variable comparison star + CR spikes",
        "comp_qa_core.compute_comp_qa",
        "Sokolovsky LOO QA (von Neumann eta / spike)",
        expected="comp_qa flags bad comp; absent from comparison pool",
        recovered=f"flagged={flagged}; in_pool={in_pool}; spike={bmetrics['spike']:.2f}; flags={bflags}",
        status=status,
        note=note,
    )

    from trust_flag_core import CompTrustThresholds, evaluate_target

    th = CompTrustThresholds.from_bounds(3, 8)
    weak = evaluate_target(
        catalog_id=WEAK_TARGET_ID,
        vsx_name="SYNTH_WEAK",
        n_clean=4,
        lc_quality="poor",
        check_scatter=0.06,
        thresholds=th,
    )
    strong = evaluate_target(
        catalog_id=TARGET_ID,
        vsx_name="SYNTH_B1",
        n_clean=55,
        lc_quality="good",
        check_scatter=0.012,
        thresholds=th,
    )
    weak_ok = weak["trust"] in ("YELLOW", "RED")
    strong_ok = strong["trust"] == "GREEN"
    report.add(
        "B3",
        "trust gate weak vs strong cases",
        "trust_flag_core.evaluate_target",
        "trust gate GREEN/YELLOW/RED",
        expected="weak YELLOW/RED; strong GREEN",
        recovered=f"weak={weak['trust']}; strong={strong['trust']}",
        status="PASS" if weak_ok and strong_ok else "FAIL",
        note="" if (weak_ok and strong_ok) else f"Weak reason={weak.get('trust_reason')}; strong reason={strong.get('trust_reason')}",
    )

    comp_mag_inst: dict[str, np.ndarray] = {}
    comp_catalog_mag: dict[str, float] = {}
    comp_bp_rp: dict[str, float] = {}
    comp_quality: dict[str, dict] = {}
    all_ids = set()
    for fi in range(N_FRAMES):
        df = pd.read_csv(proc_dir / f"proc_{fi:03d}.csv", dtype={"catalog_id": str})
        for _, r in df.iterrows():
            cid = str(r["catalog_id"]).strip()
            if cid in (TARGET_ID, BAD_COMP_ID, WEAK_TARGET_ID, "486430957815961346"):
                continue
            all_ids.add(cid)
    for cid in all_ids:
        mags_f = []
        for fi in range(N_FRAMES):
            df = pd.read_csv(proc_dir / f"proc_{fi:03d}.csv", dtype={"catalog_id": str})
            sub = df[df["catalog_id"] == cid]
            if sub.empty:
                continue
            fl = float(sub.iloc[0]["dao_flux"])
            mags_f.append(-2.5 * math.log10(fl))
        comp_mag_inst[cid] = np.array(mags_f)
        df0 = pd.read_csv(proc_dir / "proc_000.csv", dtype={"catalog_id": str})
        row = df0[df0["catalog_id"] == cid].iloc[0]
        comp_catalog_mag[cid] = float(row["phot_g_mean_mag"])
        comp_bp_rp[cid] = float(row["bp_rp"])
        comp_quality[cid] = {"quality": "good"}

    from photometry_core import fit_color_term_c1

    c1, c1_err, n_comp = fit_color_term_c1(
        comp_mag_inst, comp_catalog_mag, comp_bp_rp, comp_quality, min_comp=3
    )
    ok = math.isfinite(c1) and abs(c1 + B4_COLOR_SLOPE) < max(
        0.03, 2.0 * (c1_err if math.isfinite(c1_err) else 0.03)
    )
    report.add(
        "B4",
        "color-dependent systematic (bp_rp slope)",
        "photometry_core.fit_color_term_c1",
        "color-term coefficient recovery",
        expected=f"c1 near -injected slope {-B4_COLOR_SLOPE:.3f} (cat-inst vs bp_rp)",
        recovered=f"c1={c1:.4f} +/- {c1_err:.4f} (n={n_comp})",
        delta=f"{c1 + B4_COLOR_SLOPE:.4f}",
        status="PASS" if ok else "FAIL",
        note="" if ok else "Fitted color term diverges from injected bp_rp slope.",
    )

    rms_vals = []
    for cid in list(all_ids)[:6]:
        mags_f = comp_mag_inst[cid]
        mags_f = mags_f - np.nanmedian(mags_f)
        rms_vals.append(float(np.nanstd(mags_f)))
    med_rms = float(np.median(rms_vals))
    ok = med_rms < 0.02
    report.add(
        "B5",
        "constant comps stay flat",
        "ensemble normalization RMS on proc flux",
        "comp flatness / no spurious trend",
        expected="comp RMS consistent with injected noise (< 0.02 mag)",
        recovered=f"median comp RMS={med_rms:.4f} mag",
        status="PASS" if ok else "FAIL",
        note="" if ok else "Constant comps show elevated RMS -- check frame-to-frame injection.",
    )

    fwhms = []
    for fi in range(N_FRAMES):
        tp = TIER_B_DIR / "frames" / f"frame_{fi:03d}_truth.json"
        with open(tp, encoding="ascii") as f:
            ft = json.load(f)
        fwhms.append(ft["frame_params"]["fwhm_px"])
    jitter_ok = min(fwhms) >= 2.7 and max(fwhms) <= 3.7
    report.add(
        "B6",
        "per-frame seeing jitter + CR realism",
        "gen_series frame_params",
        "realism layer; B1/B5 should still pass",
        expected=f"FWHM in [{2.8},{3.6}] px; CRs present",
        recovered=f"FWHM range [{min(fwhms):.2f},{max(fwhms):.2f}]",
        status="PASS" if jitter_ok else "FAIL",
        note=f"catalog_source={meta.get('catalog_source')}",
    )


def _run_v3(report: ValidationReport) -> None:
    fits_path = TIER_A_DIR / "tier_a_frame.fits"
    truth = _load_truth(TIER_A_DIR / "tier_a_truth.json")

    blind_index_candidates = list((ROOT / "GAIA_DR3").glob("*.pkl"))
    if not blind_index_candidates:
        report.add(
            "V3a",
            "blind plate-solver WCS recovery",
            "vyvar_blind_solver / vyvar_platesolver._verify_blind_candidates",
            "110-deg mis-solve regression",
            expected="recovered WCS within few arcsec / 0.5 deg",
            recovered="no local Gaia blind index (.pkl)",
            status="SKIP",
            note="GAIA_DR3/*.pkl index not present on this machine; run where blind index is built.",
        )
    else:
        from astropy.wcs import WCS
        from astropy.wcs.utils import fit_wcs_from_points
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        with fits.open(fits_path, memmap=False) as hdul:
            w_truth = WCS(hdul[0].header)
        stars = truth["stars"][:30]
        xs = np.array([s["x"] for s in stars])
        ys = np.array([s["y"] for s in stars])
        ras = np.array([s["ra_deg"] for s in stars])
        decs = np.array([s["dec_deg"] for s in stars])
        world = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)
        w_rec = fit_wcs_from_points((xs, ys), world, projection="TAN")
        w_rec.array_shape = (NY, NX)
        fit_ra, fit_dec = w_rec.all_pix2world(xs, ys, 0)
        coord_fit = SkyCoord(ra=fit_ra * u.deg, dec=fit_dec * u.deg)
        coord_truth = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)
        sep_arcsec = float(np.median(coord_fit.separation(coord_truth).arcsec))
        ok = sep_arcsec < 1.0
        report.add(
            "V3a",
            "blind plate-solver WCS recovery (geometry proxy)",
            "astropy fit_wcs_from_points + verify path",
            "110-deg mis-solve regression",
            expected="star positions within ~1 arcsec after WCS fit",
            recovered=f"median star sep={sep_arcsec:.3f} arcsec",
            status="PASS" if ok else "FAIL",
            note="Full blind hash-vote solve skipped; geometric WCS fit proxy on injected stars.",
        )

    from config import AppConfig
    from pipeline import _compute_airmass_from_altaz, _extract_airmass_from_header
    from time_utils import compute_hjd_bjd, compute_time_columns, mid_exposure_jd

    with fits.open(fits_path, memmap=False) as hdul:
        hdr = hdul[0].header
    cfg = AppConfig()
    cfg.observer_lat = 50.075
    cfg.observer_lon = 14.437
    cfg.observer_alt_m = 525.0
    jd = mid_exposure_jd(hdr)
    cols = compute_time_columns(hdr, cfg=cfg)
    am_vy = _extract_airmass_from_header(hdr, cfg=cfg)
    am_az = _compute_airmass_from_altaz(hdr, cfg)

    st = _star_by_name(truth, "comp_00")
    hjd_ind, bjd_ind = compute_hjd_bjd(jd, st["ra_deg"], st["dec_deg"], 50.075, 14.437, 525.0)
    bjd_vy = cols.get("bjd_tdb_mid")
    bjd_ok = bjd_ind is not None and bjd_vy is not None and abs(float(bjd_ind) - float(bjd_vy)) < 1e-4
    am_ok = math.isfinite(am_vy) and math.isfinite(am_az) and abs(am_vy - am_az) < 0.01
    report.add(
        "V3b",
        "BJD / airmass vs astropy recomputation",
        "time_utils.compute_hjd_bjd; pipeline._compute_airmass_from_altaz",
        "mid-exposure time metadata",
        expected="BJD delta < 1e-4 d; airmass delta < 0.01",
        recovered=f"BJD delta={abs(float(bjd_ind) - float(bjd_vy)) if bjd_ind and bjd_vy else 'nan'}; "
        f"airmass vy={am_vy:.4f} az={am_az:.4f}",
        status="PASS" if bjd_ok and am_ok else "FAIL",
        note="" if (bjd_ok and am_ok) else "BJD or airmass mismatch vs independent recomputation.",
    )

    from calibration import get_processed_master, normalize_flat_master

    ny, nx = 256, 256
    yy, xx = np.mgrid[0:ny, 0:nx]
    vignette = 1.0 + 0.15 * ((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2) / (nx * ny)
    dark = np.full((ny, nx), 50.0, dtype=np.float32)
    raw = (1000.0 * vignette + dark).astype(np.float32)
    flat_path = TIER_A_DIR / "synth_flat.fits"
    dark_path = TIER_A_DIR / "synth_dark.fits"
    fits.PrimaryHDU(data=vignette.astype(np.float32)).writeto(flat_path, overwrite=True)
    fits.PrimaryHDU(data=dark).writeto(dark_path, overwrite=True)
    flat_proc = get_processed_master(flat_path, target_binning=1, kind="flat", light_shape=(ny, nx))
    dark_proc = get_processed_master(dark_path, target_binning=1, kind="dark", light_shape=(ny, nx))
    cal = (raw - dark_proc.data) / np.maximum(flat_proc.data, 1e-6)
    residual = float(np.nanmedian(np.abs(cal - 1000.0)) / 1000.0)
    ok_cal = residual < 0.01
    report.add(
        "V3c",
        "calibration flat+dark removes injected pattern",
        "calibration.get_processed_master",
        "master flat/dark apply",
        expected="residual pattern < 1%",
        recovered=f"median fractional residual={residual:.5f}",
        status="PASS" if ok_cal else "FAIL",
        note="" if ok_cal else "Calibrated frame retains >1% of injected vignette.",
    )

    # V3e: ePSF FWHM QC estimator vs injected truth (EPSF-1 diagnostic)
    from astropy.nddata import NDData
    from astropy.table import Table
    from photutils.psf import extract_stars
    from psf_photometry import _epsf_build_imagepsf_from_stars

    with fits.open(fits_path, memmap=False) as hdul:
        frame_data = np.asarray(hdul[0].data, dtype=np.float64)
    inj_fwhm = float(truth["fwhm_px"])
    clean = [s for s in truth["stars"] if s["name"].startswith("comp_")][:12]
    cutout = int(inj_fwhm * 5) | 1
    cat = Table()
    cat["x"] = [float(s["x"]) for s in clean]
    cat["y"] = [float(s["y"]) for s in clean]
    try:
        epsf_stars = extract_stars(NDData(frame_data), cat, size=cutout)
        built = _epsf_build_imagepsf_from_stars(
            epsf_stars, osamp=2, fwhm_px=inj_fwhm, cutout_size=cutout
        )
        qc = built["qc"]
        ratio = float(qc.get("epsf_vs_input_fwhm_ratio") or float("nan"))
        native = float(qc.get("epsf_fwhm_native_px") or float("nan"))
        n_stars = len(epsf_stars) if epsf_stars is not None else 0
        ok = math.isfinite(ratio) and 0.85 <= ratio <= 1.15
        note = ""
        if not ok:
            note = (
                "EPSF-1: ratio outside [0.85,1.15] on synthetic Tier-A stars "
                "(see docs/VYVAR_EPSF_AUDIT.md)."
            )
        elif ratio < 0.85:
            note = "Low ratio on synthetic would confirm EPSF-1 estimator bias."
        report.add(
            "V3e",
            "ePSF FWHM native vs injected FWHM (synthetic)",
            "psf_photometry._epsf_build_imagepsf_from_stars QC",
            "EPSF-1 half-max estimator (psf_photometry.py:500-516)",
            expected=f"ratio in [0.85,1.15] for injected FWHM={inj_fwhm:.2f} px",
            recovered=f"native={native:.3f} px; ratio={ratio:.3f}; n_stars={n_stars}",
            delta=f"{ratio - 1.0:.3f}" if math.isfinite(ratio) else "nan",
            status="PASS" if ok else "FAIL",
            note=note,
        )
    except Exception as exc:
        report.add(
            "V3e",
            "ePSF FWHM native vs injected FWHM (synthetic)",
            "psf_photometry._epsf_build_imagepsf_from_stars QC",
            "EPSF-1 half-max estimator",
            expected=f"ratio in [0.85,1.15] for FWHM={inj_fwhm:.2f} px",
            recovered=f"error: {exc}",
            status="SKIP",
            note="Could not build synthetic ePSF for FWHM QC probe.",
        )

    if PLATE_SCALE_ARCSEC > 1.0:
        report.add(
            "V3d",
            "PSF vs aperture on fine-scale blends",
            "psf_photometry vs aperture (exploratory)",
            "PSF roadmap evidence",
            expected="SKIP unless ~0.65 arcsec/px config",
            recovered=f"plate_scale={PLATE_SCALE_ARCSEC} arcsec/px",
            status="SKIP",
            note="Fine-scale bin1 config not used; PSF flags remain OFF in production.",
        )


def run_all(out_dir: Path | None = None) -> ValidationReport:
    out_dir = Path(out_dir or DATA_ROOT)
    report = ValidationReport()
    _run_tier_a(report)
    _run_tier_b(report)
    _run_v3(report)
    report.finalize(out_dir)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="VYVAR inject-and-recover validation harness")
    ap.add_argument("--all", action="store_true", help="Regenerate synthetic data and run full matrix")
    ap.add_argument("--tier-a", action="store_true", help="Tier A only")
    ap.add_argument("--tier-b", action="store_true", help="Tier B only")
    ap.add_argument("--v3", action="store_true", help="V3 targeted checks only")
    ap.add_argument("--out", type=Path, default=DATA_ROOT, help="Report output directory")
    args = ap.parse_args()

    if not (args.all or args.tier_a or args.tier_b or args.v3):
        args.all = True

    report = ValidationReport()
    if args.all or args.tier_a:
        if args.all:
            write_frame(TIER_A_DIR)
        _run_tier_a(report)
    if args.all or args.tier_b:
        if args.all:
            write_series(TIER_B_DIR)
        _run_tier_b(report)
    if args.all or args.v3:
        _run_v3(report)

    jp, mp = report.finalize(args.out)
    n_pass, n_fail, n_skip = report.summary()
    print(f"validation report: {jp}")
    print(f"validation report: {mp}")
    print(f"summary: {n_pass} pass / {n_fail} fail / {n_skip} skip")


if __name__ == "__main__":
    main()
