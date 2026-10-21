"""MERGE-MAIN-01: iter4 overlay branch must not TypeError on decompose_holes_le13."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
_ITER4_PATH = ROOT / "src_py" / "dao_gaia_stage_01_iter4.py"


def _load_src_iter4():
    """Load src_py iter4 by file path so gitignored tmp/ copies cannot shadow it."""
    sys.modules.pop("dao_gaia_stage_01_iter4", None)
    spec = importlib.util.spec_from_file_location("dao_gaia_stage_01_iter4", _ITER4_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dao_gaia_stage_01_iter4"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_frame_result(iter4, frame: str):
    from masterstar_gaia_accounting import SOURCE_TOO_FAINT

    census = pd.DataFrame(
        {
            "catalog_id": ["g1"],
            "g_mag": [12.0],
            "source_state": [SOURCE_TOO_FAINT],
            "x_gaia": [40.0],
            "y_gaia": [40.0],
        }
    )
    gaia = pd.DataFrame({"x_gaia": [40.0], "y_gaia": [40.0], "phot_g_mean_mag": [12.0]})
    data0 = np.ones((120, 120), dtype=np.float32)
    return iter4.FrameResult(
        frame=frame,
        detections=[],
        gaia_le16=gaia,
        gaia_g18=gaia,
        census=census,
        owner_kind=np.array([""], dtype=object),
        g1_strict_le13=1.0,
        g1_strict_le145=1.0,
        g1_eye_le13=1.0,
        g1_eye_le145=1.0,
        g1_eye_seed_le13=1.0,
        g1_eye_seed_le145=1.0,
        g2=0.0,
        g3_g18=0.0,
        n_det_pass1=0,
        n_det_pass2=0,
        n_forced_seed=0,
        n_ambiguous=0,
        n_crowded_miss=0,
        state_counts={SOURCE_TOO_FAINT: 1},
        g4_ok=True,
        data0=data0,
        wpx=120,
        h=120,
    )


def test_iter4_overlay_branch_accepts_single_holes_dataframe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches the P2-6 leftover TypeError: unpacking a DataFrame as (holes, summ)."""
    iter4 = _load_src_iter4()
    assert Path(iter4.__file__).resolve() == _ITER4_PATH.resolve()

    dummy_fits = tmp_path / "MASTERSTAR.fits"
    dummy_fits.write_bytes(b"x")
    sky = tmp_path / "empty_sky.csv"
    sky.write_text("x,y\n10,10\n", encoding="utf-8")
    data0 = np.ones((120, 120), dtype=np.float32)

    monkeypatch.setattr(iter4, "FRAMES", [("MASTERSTAR", dummy_fits)])
    monkeypatch.setattr(iter4, "EMPTY_SKY_CSV", sky)
    monkeypatch.setattr(
        iter4,
        "load_frame",
        lambda _path: (data0, data0, {}, None, 3.0, 120, 120),
    )
    monkeypatch.setattr(
        iter4,
        "forced_seed_empty_sky_audit",
        lambda *_a, **_k: {"n": 0},
    )
    monkeypatch.setattr(
        iter4,
        "run_frame_i6_i7",
        lambda frame_label, *_a, **_k: _fake_frame_result(iter4, frame_label),
    )
    monkeypatch.setattr(iter4, "render_overlay_final", lambda *_a, **_k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dao_gaia_stage_01_iter4.py", "--ctx", str(tmp_path / "ctx")],
    )

    iter4.main()

    holes_path = tmp_path / "ctx" / "holes_le13_final.csv"
    assert holes_path.is_file()
    holes = pd.read_csv(holes_path)
    assert not holes.empty
    assert "catalog_id" in holes.columns
    assert not (tmp_path / "ctx" / "holes_le13_decompose_final.csv").exists()
