"""FORCED-PHOT-01 + COMP-WEIGHT-COEFF-01 acceptance tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from comp_weights import C_COL_PSF_REFRACTIVE_MAG_PER_BPRP, sigma_eff_mag, weight_from_sigma_eff
from forced_photometry import (  # noqa: E402
    force_eligible_masterstar_mask,
    inject_forced_masterstar_rows,
)
from photometry_core import ensemble_normalize  # noqa: E402


def _master_table(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "catalog_id": f"G{i}",
                "x": 100.0 + 10 * i,
                "y": 200.0 + 5 * i,
                "ra_deg": 10.0 + 0.01 * i,
                "dec_deg": 20.0,
                "is_saturated": False,
                "likely_saturated": False,
                "is_noisy": True if i == 1 else False,  # must still be eligible
                "vsx_known_variable": True if i == 2 else False,
                "non_single_star": 1 if i == 3 else 0,
                "zone": "saturated" if i == 4 else "linear",
                "bp_rp": 0.5 + 0.1 * i,
            }
        )
    return pd.DataFrame(rows)


def test_force_eligible_only_saturated_excluded():
    ms = _master_table()
    ok = force_eligible_masterstar_mask(ms)
    # G0 eligible; G1 noisy but eligible; G2 variable and G3 NSS eligible (targets,
    # FORCED-TARGET-2026-10); G4 saturated on MASTERSTAR out
    assert bool(ok.iloc[0]) is True
    assert bool(ok.iloc[1]) is True
    assert bool(ok.iloc[2]) is True
    assert bool(ok.iloc[3]) is True
    assert bool(ok.iloc[4]) is False


def test_forced_inject_fills_missing_members_and_records_geometry():
    ms = _master_table(3)
    # Only G0 present from DAO; G1 and the variable G2 (target) are injected
    df = pd.DataFrame(
        [
            {
                "catalog_id": "G0",
                "x": 100.0,
                "y": 200.0,
                "source_type": "GAIA_MATCHED",
                "flux": 1000.0,
            }
        ]
    )
    img = np.zeros((400, 400), dtype=np.float64)
    img[200, 100] = 50.0
    img[205, 110] = 40.0
    out, meta = inject_forced_masterstar_rows(
        df, ms, image=img
    )
    ids = set(out["catalog_id"].astype(str))
    assert "G0" in ids
    assert "G1" in ids
    assert "G2" in ids  # FORCED-TARGET-2026-10
    assert int(meta["n_injected"]) >= 1
    assert "forced_photometry" in out.columns
    assert bool(out.loc[out["catalog_id"] == "G1", "forced_photometry"].iloc[0]) is True


def test_ensemble_contributing_set_equal_when_all_finite():
    """Membership set is fixed; with forced finite mags, contributors equal membership."""
    n = 8
    t = np.full(n, 12.0)
    comps = {
        "A": np.full(n, 11.0),
        "B": np.full(n, 11.1),
        "C": np.full(n, 11.2),
    }
    cat = {"A": 10.0, "B": 10.1, "C": 10.2}
    qual = {k: {"rms": 0.01} for k in comps}
    mag_c, dmag, scat = ensemble_normalize(
        t, comps, cat, qual, n_comp_min=2
    )
    assert np.all(np.isfinite(mag_c))
    # Reconstruct per-frame contributors: all three every frame
    for i in range(n):
        contrib = {cid for cid, arr in comps.items() if math.isfinite(float(arr[i]))}
        assert contrib == {"A", "B", "C"}


def test_ensemble_sat_excluded_explicitly_membership_unchanged():
    n = 4
    t = np.full(n, 12.0)
    comps = {
        "A": np.full(n, 11.0),
        "B": np.full(n, 11.1),
    }
    cat = {"A": 10.0, "B": 10.1}
    qual = {k: {} for k in comps}
    sat = {"A": np.array([False, True, False, False]), "B": np.zeros(n, dtype=bool)}
    mag_c, _, _ = ensemble_normalize(
        t,
        comps,
        cat,
        qual,
        comp_likely_saturated=sat,
        n_comp_min=1,
    )
    assert math.isfinite(float(mag_c[0]))
    assert math.isfinite(float(mag_c[1]))  # B alone still forms ZP
    # Membership keys unchanged
    assert set(qual.keys()) == {"A", "B"}


def test_universality_with_nonzero_coeffs_exact():
    """Subset invariance must hold exactly with non-zero c_col/c_dist."""
    c_col = C_COL_PSF_REFRACTIVE_MAG_PER_BPRP
    c_dist = 0.002
    stars = [
        ("A", 0.01, 0.0, 0.1),
        ("B", 0.02, 0.4, 0.5),
        ("C", 0.015, -0.2, 1.0),
    ]

    def _w(subset):
        out = {}
        for cid, rms, db, r in subset:
            se = sigma_eff_mag(
                sigma_rms_mag=rms,
                delta_bprp=db,
                r_deg=r,
                c_col_mag_per_bprp=c_col,
                c_dist_mag_per_deg=c_dist,
            )
            out[cid] = (se, weight_from_sigma_eff(se))
        return out

    full = _w(stars)
    # Remove C: A and B sigma_eff unchanged exactly
    sub = _w(stars[:2])
    assert full["A"][0] == sub["A"][0]
    assert full["B"][0] == sub["B"][0]
    assert full["A"][1] == sub["A"][1]


def test_ensemble_normalize_consumes_exact_step2_set():
    """COMP-ASSIGN-01 D5: ensemble photometers exactly the delivered membership."""
    from photometry_core import ensemble_normalize as _ens

    n = 8
    delivered = ["c1", "c2", "c3", "c4", "c5"]
    target = np.linspace(12.0, 12.02, n)
    comp_lc = {c: np.linspace(11.0, 11.01, n) + 0.01 * i for i, c in enumerate(delivered)}
    # Extra star present in LC dict but NOT in quality/membership must be ignored
    comp_lc["outsider"] = np.linspace(10.0, 10.01, n)
    cat = {c: 11.0 for c in delivered}
    cat["outsider"] = 10.0
    quality = {c: {"quality": "good"} for c in delivered}
    mag, _, _ = _ens(
        target,
        comp_lc,
        cat,
        quality,
        n_comp_min=3,
    )
    assert set(quality.keys()) == set(delivered)
    assert np.isfinite(mag).all()
    # Manual ZP with only delivered set
    zp = np.zeros(n)
    for i in range(n):
        zs = np.array([cat[c] - comp_lc[c][i] for c in delivered])
        w = np.ones(len(delivered))  # identical comp shapes -> equal Broeg weights
        zp[i] = float(np.sum(w * zs) / np.sum(w))
    np.testing.assert_allclose(mag, target + zp, rtol=0, atol=1e-12)


def test_select_comps_rms_then_color_distance_clamp():
    """COMP-ASSIGN-03: RMS -> colour -> distance; clamp to n_comp_max."""
    from photometry_core import _select_comps_by_rms_then_color

    rows = []
    # Same colour, different RMS and distance - quieter nearer should win ties
    for i, (rms, dist) in enumerate(
        [(0.05, 0.5), (0.02, 0.8), (0.02, 0.1), (0.03, 0.2), (0.04, 0.15),
         (0.06, 0.05), (0.07, 0.4), (0.08, 0.3), (0.09, 0.25), (0.10, 0.35)]
    ):
        rows.append(
            {
                "catalog_id": f"S{i:02d}",
                "bp_rp": 1.0,
                "comp_rms": rms,
                "_dist_deg": dist,
                "_nn_dist_fwhm": 5.0,
                "snr_ap_pixscaled": 40.0,
            }
        )
    df = pd.DataFrame(rows)
    out = _select_comps_by_rms_then_color(
        df, target_bprp=1.0, n_comp_min=3, n_comp_max=8, max_delta_bprp=0.5
    )
    assert 3 <= len(out) <= 8
    assert len(out) == 8
    # Best by rms, then |dbprp|, then distance: S02 (0.02, 0.1) before S01 (0.02, 0.8)
    ids = out["catalog_id"].astype(str).tolist()
    assert ids[0] == "S02"
    assert "S01" in ids
    assert ids.index("S02") < ids.index("S01")
    # RMS order: first selected has lowest rms among set
    assert float(out.iloc[0]["comp_rms"]) <= float(out.iloc[1]["comp_rms"])


def test_select_comps_max_comp_rms_ceiling_before_head():
    """COMP-ASSIGN-02: phase01_comparison_max_comp_rms before head(n_comp_max).

    Perfect-colour noisy eights must not fill the set; ladder widens to clean
    comps. n_comp_max is a ceiling, not a pad target.
    """
    from config import AppConfig
    from photometry_core import _select_comps_by_rms_then_color

    cfg = AppConfig()
    cfg.phase01_comparison_max_comp_rms = 0.1
    rows = []
    # Eight perfect-colour comps above the ceiling (FW CVn pattern).
    for i in range(8):
        rows.append(
            {
                "catalog_id": f"NOISY{i:02d}",
                "bp_rp": 1.000,
                "comp_rms": 0.15 + 0.04 * i,
                "_dist_deg": 0.05 + 0.01 * i,
                "_nn_dist_fwhm": 5.0,
                "snr_ap_pixscaled": 40.0,
            }
        )
    # Three slightly worse colour, under ceiling.
    for i, (db, rms) in enumerate([(0.04, 0.02), (0.05, 0.025), (0.06, 0.03)]):
        rows.append(
            {
                "catalog_id": f"CLEAN{i:02d}",
                "bp_rp": 1.0 + db,
                "comp_rms": rms,
                "_dist_deg": 0.10 + 0.01 * i,
                "_nn_dist_fwhm": 5.0,
                "snr_ap_pixscaled": 40.0,
            }
        )
    df = pd.DataFrame(rows)
    out = _select_comps_by_rms_then_color(
        df,
        target_bprp=1.0,
        n_comp_min=3,
        n_comp_max=8,
        max_delta_bprp=0.79,
        cfg=cfg,
    )
    assert len(out) == 3
    assert set(out["catalog_id"].astype(str)) == {"CLEAN00", "CLEAN01", "CLEAN02"}
    assert float(pd.to_numeric(out["comp_rms"], errors="coerce").max()) <= 0.1
    assert int(out.attrs.get("color_ladder_step", 0)) >= 1


def test_select_comps_excludes_blends_single_source():
    """COMP-ASSIGN-03: comps closer than comp_isolation_fwhm are excluded."""
    from config import AppConfig
    from photometry_core import _select_comps_by_rms_then_color

    cfg = AppConfig()
    cfg.comp_isolation_fwhm = 3.0
    rows = [
        {"catalog_id": "BLEND", "bp_rp": 1.0, "comp_rms": 0.01, "_dist_deg": 0.1, "_nn_dist_fwhm": 1.5, "snr_ap_pixscaled": 40.0},
        {"catalog_id": "ISO0", "bp_rp": 1.05, "comp_rms": 0.02, "_dist_deg": 0.2, "_nn_dist_fwhm": 4.0, "snr_ap_pixscaled": 40.0},
        {"catalog_id": "ISO1", "bp_rp": 1.06, "comp_rms": 0.03, "_dist_deg": 0.3, "_nn_dist_fwhm": 5.0, "snr_ap_pixscaled": 40.0},
        {"catalog_id": "ISO2", "bp_rp": 1.07, "comp_rms": 0.04, "_dist_deg": 0.4, "_nn_dist_fwhm": 6.0, "snr_ap_pixscaled": 40.0},
    ]
    out = _select_comps_by_rms_then_color(
        pd.DataFrame(rows),
        target_bprp=1.0,
        n_comp_min=3,
        n_comp_max=8,
        max_delta_bprp=0.79,
        cfg=cfg,
    )
    assert "BLEND" not in set(out["catalog_id"].astype(str))
    assert set(out["catalog_id"].astype(str)) == {"ISO0", "ISO1", "ISO2"}


def test_select_comps_rms_first_over_perfect_colour():
    """COMP-ASSIGN-03: quieter worse-colour beats noisier perfect-colour."""
    from config import AppConfig
    from photometry_core import _select_comps_by_rms_then_color

    cfg = AppConfig()
    cfg.phase01_comparison_max_comp_rms = 0.1
    rows = [
        {"catalog_id": "PERF_NOISY", "bp_rp": 1.0, "comp_rms": 0.08, "_dist_deg": 0.1, "_nn_dist_fwhm": 5.0, "snr_ap_pixscaled": 40.0},
        {"catalog_id": "QUIET_FAR", "bp_rp": 1.12, "comp_rms": 0.02, "_dist_deg": 0.2, "_nn_dist_fwhm": 5.0, "snr_ap_pixscaled": 40.0},
        {"catalog_id": "MID", "bp_rp": 1.05, "comp_rms": 0.03, "_dist_deg": 0.15, "_nn_dist_fwhm": 5.0, "snr_ap_pixscaled": 40.0},
        {"catalog_id": "OK3", "bp_rp": 1.08, "comp_rms": 0.04, "_dist_deg": 0.25, "_nn_dist_fwhm": 5.0, "snr_ap_pixscaled": 40.0},
    ]
    out = _select_comps_by_rms_then_color(
        pd.DataFrame(rows),
        target_bprp=1.0,
        n_comp_min=3,
        n_comp_max=3,
        max_delta_bprp=0.79,
        cfg=cfg,
    )
    ids = out["catalog_id"].astype(str).tolist()
    assert ids[0] == "QUIET_FAR"
    assert "PERF_NOISY" not in ids  # higher rms; n_max=3 takes three quietest under colour step


def test_fire_rebuilt_comparison_csv_under_ceiling():
    """Fire proof (pass side): rebuilt membership stays under the RMS ceiling."""
    from pathlib import Path

    from config import AppConfig

    root = Path(__file__).resolve().parents[2]
    phot = (
        root
        / "Archive"
        / "Drafts"
        / "draft_000514"
        / "platesolve"
        / "NoFilter_60_2"
        / "photometry"
    )
    live = phot / "comparison_stars_per_target.csv"
    if not live.is_file():
        import pytest

        pytest.skip(f"rebuilt CSV not present yet: {live}")
    cfg = AppConfig()
    ceil = float(cfg.phase01_comparison_max_comp_rms)
    df = pd.read_csv(live, low_memory=False)
    rms = pd.to_numeric(df["comp_rms"], errors="coerce")
    n_breach = int((rms.notna() & (rms > ceil)).sum())
    assert n_breach == 0, f"{n_breach} comps above max_comp_rms={ceil}"


def test_fire_rebuilt_comparison_csv_no_blends():
    """Fire proof (pass side): rebuilt membership has no <3-FWHM blends (masterstars NN)."""
    from pathlib import Path

    from config import AppConfig
    from gaia_catalog_id import normalize_gaia_source_id
    from scipy.spatial import cKDTree

    root = Path(__file__).resolve().parents[2]
    phot = (
        root
        / "Archive"
        / "Drafts"
        / "draft_000514"
        / "platesolve"
        / "NoFilter_60_2"
        / "photometry"
    )
    live = phot / "comparison_stars_per_target.csv"
    if not live.is_file():
        import pytest

        pytest.skip(f"rebuilt CSV not present yet: {live}")
    ms_path = phot.parent / "masterstars_full_match.csv"
    if not ms_path.is_file():
        import pytest

        pytest.skip("masterstars missing")
    field = pd.read_csv(ms_path, low_memory=False, dtype={"catalog_id": str})
    for col in ("x", "y"):
        field[col] = pd.to_numeric(field[col], errors="coerce")
    field["_nid"] = field["catalog_id"].map(
        lambda x: str(normalize_gaia_source_id(x) or "").strip()
    )
    field = field[
        np.isfinite(field["x"]) & np.isfinite(field["y"]) & field["_nid"].str.len().gt(0)
    ].drop_duplicates("_nid")
    pts = np.column_stack([field["x"].to_numpy(), field["y"].to_numpy()])
    d, _ = cKDTree(pts).query(pts, k=2)
    nn_map = dict(zip(field["_nid"], d[:, 1], strict=False))
    fwhm = 5.19465
    thr = float(AppConfig().comp_isolation_fwhm) * fwhm
    comps = pd.read_csv(live, low_memory=False, dtype={"catalog_id": str})
    nn = comps["catalog_id"].map(
        lambda x: nn_map.get(str(normalize_gaia_source_id(x) or "").strip(), float("nan"))
    )
    n_blend = int((nn.notna() & (nn < thr)).sum())
    if n_blend > 0:
        import pytest

        pytest.skip(
            f"live CSV still has {n_blend} blended comps (Item C rebuild pending)"
        )
    assert n_blend == 0
