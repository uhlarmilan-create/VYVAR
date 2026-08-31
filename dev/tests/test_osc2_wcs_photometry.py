# -*- coding: ascii -*-
"""OSC-2: WCS solve-once, registration handoff, unified QC, OSC-02."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from osc_align import (
    OSC_REGISTRATION_HANDOFF,
    apply_registration_handoff_to_frame,
    merge_osc_qc_metrics_at_lights_root,
    obs_group_band_token,
    parse_osc_channel_from_setup,
    partition_jobs_for_osc_alignment,
    replicate_qc_verdict_from_one_rggb,
    write_registration_handoff,
)
from osc_extract import channel_obs_group_folder, effective_gain_rn
from database import VyvarDatabase
from param_resolver import resolve_gain, resolve_read_noise
from photometry_core import (
    _howell_variance_adu2,
    _photometric_error_with_bkg_mode,
    read_flux_from_csv,
    run_full_photometry_pipeline,
    select_comparison_stars_per_target,
)
from config import AppConfig


def _write_qc_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_qc_verdict_replication_and_merge(tmp_path: Path) -> None:
    root = tmp_path / "lights"
    base = "NoFilter_60_2"
    names = ["f001.fits", "f002.fits"]
    one_dir = root / channel_obs_group_folder(base, "oneRGGB")
    r_dir = root / channel_obs_group_folder(base, "R")
    for d in (one_dir, r_dir, root / channel_obs_group_folder(base, "G"), root / channel_obs_group_folder(base, "B")):
        d.mkdir(parents=True)
    one_rows = [
        {"src": str(one_dir / n), "dst": str(one_dir / n), "status": "ok" if i == 0 else "rejected_prefilter_fwhm", "fwhm_px": 2.1 + i}
        for i, n in enumerate(names)
    ]
    _write_qc_csv(one_dir / "qc_metrics.csv", one_rows)
    for ch in ("R", "G", "B"):
        ch_dir = root / channel_obs_group_folder(base, ch)
        rows = [
            {"src": str(ch_dir / n), "dst": str(ch_dir / n), "status": "ok", "fwhm_px": 3.0 + i}
            for i, n in enumerate(names)
        ]
        _write_qc_csv(ch_dir / "qc_metrics.csv", rows)
    stat = replicate_qc_verdict_from_one_rggb(lights_root=root, base_name=base)
    assert stat["channels"]["R"]["statuses"] == ["ok", "rejected_prefilter_fwhm"]
    r_df = pd.read_csv(r_dir / "qc_metrics.csv")
    assert list(r_df["status"]) == ["ok", "rejected_prefilter_fwhm"]
    assert list(r_df["qc_source"]) == ["oneRGGB", "oneRGGB"]
    merged = merge_osc_qc_metrics_at_lights_root(root)
    assert merged is not None and merged.is_file()
    mdf = pd.read_csv(merged)
    assert len(mdf) == 8


def test_partition_ordering_and_fail_closed() -> None:
    jobs = [{"gkey": "NoFilter_60_2_R", "files": []}]
    with pytest.raises(RuntimeError, match="oneRGGB"):
        partition_jobs_for_osc_alignment(jobs)
    full = [
        {"gkey": "NoFilter_60_2_B", "files": [Path("b.fits")]},
        {"gkey": "NoFilter_60_2_oneRGGB", "files": [Path("o.fits")]},
        {"gkey": "NoFilter_60_2_R", "files": [Path("r.fits")]},
        {"gkey": "NoFilter_60_2_G", "files": [Path("g.fits")]},
        {"gkey": "Mono_30_1", "files": [Path("m.fits")]},
    ]
    ordered, meta = partition_jobs_for_osc_alignment(full)
    assert meta["has_osc_bundles"]
    assert ordered[0]["gkey"].endswith("_oneRGGB")
    assert ordered[-1]["gkey"] == "Mono_30_1"


def test_osc02_unified_frame_sets_pass_and_fail() -> None:
    from invariants_runtime import InvariantViolation, check_osc02_unified_frame_sets

    bundle = {
        "NoFilter_60_2": {
            "oneRGGB": {"files": [Path("a.fits"), Path("b.fits")]},
            "R": {"files": [Path("a.fits"), Path("b.fits")]},
            "G": {"files": [Path("a.fits"), Path("b.fits")]},
            "B": {"files": [Path("a.fits"), Path("b.fits")]},
        }
    }
    check_osc02_unified_frame_sets(bundle, meta={"invariants": []})
    bad = {
        "NoFilter_60_2": {
            "oneRGGB": {"files": [Path("a.fits"), Path("b.fits")]},
            "R": {"files": [Path("a.fits")]},
            "G": {"files": [Path("a.fits"), Path("b.fits")]},
            "B": {"files": [Path("a.fits"), Path("b.fits")]},
        }
    }
    with pytest.raises(InvariantViolation, match="OSC-02"):
        check_osc02_unified_frame_sets(bad, meta={"invariants": []})


def test_registration_handoff_known_shift() -> None:
    import astroalign

    rng = np.random.default_rng(0)
    h, w = 64, 64
    ref = rng.normal(100.0, 5.0, (h, w)).astype(np.float32)
    dx, dy = 2.0, -1.0
    from scipy.ndimage import shift as ndimage_shift

    src = ndimage_shift(ref, shift=[dy, dx], mode="nearest", order=1, prefilter=False).astype(np.float32)
    yy, xx = np.mgrid[8:56:8, 8:56:8]
    ref_pts = np.column_stack([yy.ravel(), xx.ravel()]).astype(np.float32)
    src_pts = ref_pts + np.array([dy, dx], dtype=np.float32)
    t, _ = astroalign.find_transform(source=src_pts, target=ref_pts, max_control_points=12)
    ref_hdr = fits.Header()
    src_hdr = fits.Header()
    entry = {
        "method": "astroalign",
        "matrix": [float(x) for x in np.asarray(t.params, dtype=np.float64).reshape(-1)],
        "aligned": True,
    }
    aligned, _, method = apply_registration_handoff_to_frame(
        frame_path=Path("f.fits"),
        frame_data=src,
        frame_hdr=src_hdr,
        ref_data=ref,
        ref_hdr=ref_hdr,
        handoff_entry=entry,
    )
    assert "handoff" in method
    diff_raw = float(np.nanmean(np.abs(src - ref)))
    diff_aligned = float(np.nanmean(np.abs(aligned - ref)))
    assert diff_aligned < diff_raw
    assert diff_aligned == pytest.approx(0.0, abs=6.0)


def test_write_load_registration_handoff(tmp_path: Path) -> None:
    ps = tmp_path / "ps"
    ps.mkdir()
    write_registration_handoff(
        ps,
        reference_file="ref.fits",
        frames={"f1.fits": {"method": "astroalign", "matrix": [1, 0, 0, 1, 0, 0], "aligned": True}},
    )
    payload = json.loads((ps / OSC_REGISTRATION_HANDOFF).read_text(encoding="ascii"))
    assert payload["reference_file"] == "ref.fits"
    assert "f1.fits" in payload["frames"]


@pytest.mark.parametrize(
    ("obs_group", "token"),
    [
        ("NoFilter_60_2_R", "TR"),
        ("NoFilter_60_2_G", "TG"),
        ("NoFilter_60_2_B", "TB"),
        ("NoFilter_60_2_oneRGGB", "CLEAR"),
        ("NoFilter_60_2", "NoFilter"),
    ],
)
def test_band_tokens(obs_group: str, token: str) -> None:
    assert obs_group_band_token(obs_group) == token
    base, ch = parse_osc_channel_from_setup(obs_group)
    if ch:
        assert obs_group.endswith(f"_{ch}")


def _synthetic_dense_comp_fixture() -> tuple[pd.Series, pd.DataFrame, list[Path], dict[str, pd.DataFrame]]:
    """Dense synthetic field (30 stars, 20 frames) for comp-tier smoke."""
    rng = np.random.default_rng(7)
    ids = [f"G{i:03d}" for i in range(1, 31)]
    rows: list[dict[str, object]] = []
    for i, sid in enumerate(ids):
        ang = (i + 1) * 0.035
        mag = 11.5 + (i % 8) * 0.12
        flux = 80_000.0 + i * 900.0
        rows.append(
            {
                "catalog_id": sid,
                "name": sid,
                "source_id": sid,
                "ra_deg": 180.0 + ang,
                "dec_deg": 45.0 + ang * 0.25,
                "x": 120.0 + (i % 10) * 48.0,
                "y": 120.0 + (i // 10) * 42.0,
                "phot_g_mean_mag": mag,
                "mag": mag,
                "bp_rp": 0.85 + (i % 5) * 0.05,
                "dao_flux": flux,
                "flux": flux,
                "is_usable": True,
                "is_saturated": False,
                "is_noisy": False,
                "snr50_ok": True,
                "vsx_known_variable": False,
                "likely_saturated": False,
                "zone": "linear",
                "source_state": "DETECTED_P1",
                "vy_identity_gate": "ok",
                "gaia_dao_resid_px": 0.2,
                "snr_ap_pixscaled": 80.0,
            }
        )
    masterstars = pd.DataFrame(rows)
    target = masterstars.loc[masterstars["catalog_id"] == "G005"].iloc[0]
    flux_by_id = masterstars.set_index("catalog_id", drop=False)
    per_frame_paths: list[Path] = []
    csv_cache: dict[str, pd.DataFrame] = {}
    for fi in range(20):
        path = Path(f"synthetic_proc_{fi:03d}.csv")
        per_frame_paths.append(path)
        frame_rows: list[dict[str, object]] = []
        for sid in sorted(ids):
            star = flux_by_id.loc[sid]
            base = float(star["dao_flux"])
            flux = base * (1.0 + 1e-5 * float(rng.standard_normal()))
            frame_rows.append(
                {
                    "name": sid,
                    "catalog_id": sid,
                    "bjd_tdb_mid": 2_459_000.0 + fi * 0.01,
                    "dao_flux": flux,
                    "flux": flux,
                    "mag": float(star["mag"]),
                    "noise_floor_adu": 35.0,
                    "aperture_r_px": 7.0,
                    "is_usable": True,
                    "is_saturated": False,
                    "is_noisy": False,
                    "snr50_ok": True,
                    "vsx_known_variable": False,
                    "likely_saturated": False,
                }
            )
        csv_cache[str(path)] = pd.DataFrame(frame_rows)
    return target, masterstars, per_frame_paths, csv_cache


def test_production_path_howell_variance_uses_vy_egain_rdnois(tmp_path: Path) -> None:
    """Phase-2A path inside run_full_photometry_pipeline: read_flux_from_csv -> _howell_variance_adu2."""
    assert callable(run_full_photometry_pipeline)
    g_raw, rn_raw = 1.5, 3.0
    g_eff, rn_eff = effective_gain_rn(g_raw, rn_raw, "G", 2)
    ch_ms = tmp_path / "MASTERSTAR_G.fits"
    ch_hdr = fits.Header()
    ch_hdr["VY_EGAIN"] = (float(g_eff), "Effective gain e-/ADU (VYVAR OSC)")
    ch_hdr["VY_RDNOIS"] = (float(rn_eff), "Effective read noise e- (VYVAR OSC)")
    ch_hdr["VY_CHANNEL"] = ("G", "OSC channel")
    ch_hdr["EGAIN"] = (g_raw, "Raw gain e-/ADU")
    ch_hdr["RDNOISE"] = (rn_raw, "Raw read noise e-")
    ch_hdr["VY_FWHM"] = (3.2, "FWHM px")
    fits.writeto(ch_ms, np.ones((16, 16), dtype=np.float32), ch_hdr, overwrite=True)
    with fits.open(ch_ms, memmap=False) as hdul:
        hdr = hdul[0].header
    db_path = Path(__file__).resolve().parents[2] / "vyvar.sqlite3"
    if not db_path.is_file():
        pytest.skip("vyvar.sqlite3 not available for production-path gain/RN resolution")
    db = VyvarDatabase(db_path)
    g_res = resolve_gain(hdr, db=db, equipment_id=2)
    rn_res = resolve_read_noise(hdr, db=db, equipment_id=2)
    assert g_res.ok and rn_res.ok
    assert float(g_res.value) == pytest.approx(g_eff)
    assert g_res.source == "header"
    assert g_res.key == "VY_EGAIN"
    rn_hdr = resolve_read_noise(hdr)
    assert rn_hdr.ok
    assert float(rn_hdr.value) == pytest.approx(rn_eff)
    assert rn_hdr.key == "VY_RDNOIS"

    flux = 12_000.0
    sky_pp = 40.0
    area = math.pi * 7.0 * 7.0
    proc = tmp_path / "proc_001.csv"
    pd.DataFrame(
        [
            {
                "catalog_id": "G005",
                "name": "G005",
                "x": 100.0,
                "y": 100.0,
                "dao_flux": flux,
                "aperture_r_px": 7.0,
                "noise_floor_adu": sky_pp,
                "peak_max_adu": 100.0,
            }
        ]
    ).to_csv(proc, index=False)
    lc = read_flux_from_csv(
        proc,
        ["G005"],
        {"G005": 7.0},
        csv_df=pd.read_csv(proc),
        gain=float(g_res.value),
        read_noise=float(rn_hdr.value),
        err_background_mode="howell",
    )
    assert len(lc) == 1
    var_target = _howell_variance_adu2(flux, sky_pp, area, gain=g_eff, read_noise=rn_eff)
    err_target, _ = _photometric_error_with_bkg_mode(
        flux,
        err_background_mode="howell",
        sky_pp=sky_pp,
        area=area,
        gain=g_eff,
        read_noise=rn_eff,
    )
    assert float(lc.iloc[0]["err"]) == pytest.approx(err_target)
    assert (math.sqrt(var_target) / flux) == pytest.approx(err_target)

    var_mono = _howell_variance_adu2(flux, sky_pp, area, gain=g_raw, read_noise=1.3)
    assert var_mono > var_target


@pytest.mark.parametrize("channel", ["oneRGGB", "R", "G", "B"])
def test_osc_comp_selection_smoke_per_channel(channel: str) -> None:
    """Per-channel comp tiers resolve on synthetic dense field (Phase 1 machinery unchanged)."""
    target, masterstars, paths, cache = _synthetic_dense_comp_fixture()
    obs_group = channel_obs_group_folder("NoFilter_60_2", channel)
    assert obs_group.endswith(f"_{channel}")
    assert obs_group_band_token(obs_group) in {"TR", "TG", "TB", "CLEAR"}
    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    out = select_comparison_stars_per_target(
        target,
        masterstars,
        paths,
        csv_cache=cache,
        cfg=cfg,
        chip_fw=800,
        chip_fh=600,
        chip_interior_margin_px=0,
        max_psf_chi2=float("inf"),
        max_fwhm_factor=float("inf"),
        min_dist_arcsec=30.0,
        variable_target_catalog_ids=frozenset({"G005"}),
    )
    assert not out.empty
    assert "catalog_id" in out.columns
    assert "bp_rp" in out.columns or "mag" in out.columns
    assert len(out) >= 3
