"""TARGET-RULES-2026-10 (Milan 2026-10-13): Phase-0 target selection follows exactly the rules.

A target is a VSX star DAO detected on MASTERSTAR (a catalogue-membership row is a catalogue
row, not image data). Then three final cuts: saturated (upper), S/N below snr_min (lower,
drowned in noise), VSX type out of scope (ROT). Draft 532: 32 catalogue rows (CS Cam G 4.1
saturated on MASTERSTAR among them) and 11 saturated targets went into Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import AppConfig
from photometry_core import FINAL_SKIP_REASONS, select_active_targets


def _ms_row(cid: str, x: float, *, zone: str = "linear", flux: float | None = 5.0e4, snr: float | None = 200.0,
            state: str = "DETECTED_P1") -> dict:
    return {"name": cid, "catalog_id": cid, "ra_deg": 150.0 + x / 1e4, "dec_deg": 45.0, "x": x, "y": 300.0,
            "mag": 11.0, "zone": zone, "flux": flux, "snr_ap_pixscaled": snr, "source_state": state}


def _vt_row(cid: str, x: float, vsx_type: str = "EA") -> dict:
    return {"name": cid, "vsx_name": f"V{cid[-3:]}", "vsx_type": vsx_type, "vsx_period": 1.0, "priority": 1,
            "ra_deg": 150.0 + x / 1e4, "dec_deg": 45.0, "x": x, "y": 300.0, "catalog_id": cid,
            "catalog": "VSX", "gaia_match_source": "masterstars", "target_origin": "vsx_auto"}


def _run(tmp_path: Path, rows: list[tuple[dict, dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import photometry_comp

    ms_p = tmp_path / "masterstars_full_match.csv"
    vt_p = tmp_path / "variable_targets.csv"
    pd.DataFrame([m for m, _ in rows]).to_csv(ms_p, index=False)
    pd.DataFrame([v for _, v in rows]).to_csv(vt_p, index=False)
    cfg = AppConfig()
    cfg.snr_min = 10.0
    cfg.vsx_out_of_scope_types = ["ROT"]
    out = select_active_targets(vt_p, ms_p, frame_w_px=1000, frame_h_px=1000, edge_margin_px=10, cfg=cfg)
    return out.set_index("catalog_id"), photometry_comp.LAST_EXCLUDED_TARGETS


def test_rules(tmp_path: Path) -> None:
    ids = [str(1000000000000000000 + i) for i in range(8)]
    rows = [
        (_ms_row(ids[0], 100), _vt_row(ids[0], 100)),                                   # measured
        (_ms_row(ids[1], 150, zone="saturated", snr=900.0), _vt_row(ids[1], 150)),      # upper cut
        (_ms_row(ids[2], 200, snr=6.0), _vt_row(ids[2], 200)),                          # lower cut
        (_ms_row(ids[3], 250, snr=None), _vt_row(ids[3], 250)),                         # no S/N -> lower cut
        (_ms_row(ids[4], 300), _vt_row(ids[4], 300, vsx_type="ROT")),                   # out of scope
        (_ms_row(ids[5], 350, zone="unknown", flux=None, snr=None, state="catalog_membership"),
         _vt_row(ids[5], 350)),                                                         # catalogue row
        (_ms_row(ids[6], 400, zone="noise", snr=25.0), _vt_row(ids[6], 400)),           # S/N decides, not zone
        (_ms_row(ids[7], 450, zone="unknown", flux=None, snr=None, state=""), _vt_row(ids[7], 450)),  # no flux
    ]
    at, exc = _run(tmp_path, rows)
    assert not at.loc[ids[0], "skip_photometry"]
    assert at.loc[ids[1], "skip_reason"] == "saturated"
    assert at.loc[ids[2], "skip_reason"] == "below_snr_min"
    assert at.loc[ids[3], "skip_reason"] == "below_snr_min"
    assert at.loc[ids[4], "skip_reason"] == "vsx_type_out_of_scope"
    assert not at.loc[ids[6], "skip_photometry"]
    assert ids[5] not in at.index and ids[7] not in at.index
    assert set(exc.loc[exc["reason"] == "no_dao_detection", "name"].astype(str)) == {ids[5], ids[7]}
    # every Phase-0 cut is final: no comparison selection, never revived in Phase 2A
    assert {"saturated", "below_snr_min", "vsx_type_out_of_scope"} <= set(FINAL_SKIP_REASONS)
    measured = at[~at["skip_reason"].astype(str).isin(FINAL_SKIP_REASONS)]
    assert set(measured.index) == {ids[0], ids[6]}


def test_removed_paths_stay_removed() -> None:
    src = (Path(__file__).resolve().parents[2] / "src_py")
    pc = (src / "photometry_comp.py").read_text(encoding="utf-8")
    assert "_fill_missing_master_snr" not in pc and "target_depth_g" not in pc and "snr50_ok_for_skip" not in pc
    lc = (src / "photometry_lightcurve.py").read_text(encoding="utf-8")
    assert "pfs_rescue_eligible" not in lc and "decide_target_saturation_policy" not in lc
    p01 = (src / "phase01_run.py").read_text(encoding="utf-8")
    assert "target_depth.json" not in p01
