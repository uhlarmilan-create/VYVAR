"""SAT-PRECAL-2026-10 / FITS-SCALE-2026-10 (Milan 2026-10-13).

Draft 532 (QHY294, lights used without VYVAR calibration, ``non_calibrated/lights``):
- the in-place QC write kept BZERO=32768 of the 16-bit originals on float data -> +32768 ADU
  per write (sky 1664 -> 34 100 -> 66 800 on a re-run);
- SAT-DIAG looked only in ``Raw/lights`` -> no saturation level -> INV-SAT-01 stopped the run;
- float frames were refused, MASTERSTAR zones and the per-frame target test used a guessed
  65535 clip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

from pipeline_catalog import (  # noqa: E402
    _detect_empirical_clip_level_adu,
    _resolve_peak_saturation_limit_adu,
)
from sat_diag import inv_sat_limit_peak_test_adu  # noqa: E402
from pipeline_constants import LINEARITY_FRAC  # noqa: E402
from sat_diag import (  # noqa: E402
    derive_ceiling_from_paths,
    draft_peak_test_adu,
    run_sat_diag,
    sample_light_paths,
    write_sat_diag_json,
)
from utils import set_float_image_data_in_place  # noqa: E402

SAT = 65500.0


def _star_field(rng: np.random.Generator, *, n_sat: int, offset: float = 0.0) -> np.ndarray:
    h, w = 200, 300
    img = rng.normal(1500.0, 20.0, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(40):
        x, y = rng.uniform(10, w - 10), rng.uniform(10, h - 10)
        amp = 200000.0 if i < n_sat else rng.uniform(2000, 30000)
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.2**2))
    # saturated pixels clip at SAT with a small per-pixel spread (dark / sky surface subtraction)
    clip = SAT + rng.normal(0.0, 15.0, img.shape)
    img = np.minimum(img, clip)
    return (img + offset).astype(np.float32)


def _write_lights(root: Path, n: int, *, n_sat: int, offset: float = 0.0, seed: int = 1) -> list[Path]:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = root / f"L_{i:03d}.fits"
        fits.PrimaryHDU(_star_field(rng, n_sat=n_sat, offset=offset)).writeto(p, overwrite=True)
        out.append(p)
    return out


def test_in_place_float_write_keeps_values(tmp_path: Path) -> None:
    p = tmp_path / "u16.fits"
    fits.PrimaryHDU((np.arange(16, dtype=np.uint16).reshape(4, 4) + 1000)).writeto(p)
    for _ in range(3):
        x = fits.getdata(p).astype(np.float32)
        with fits.open(p, mode="update") as f:
            set_float_image_data_in_place(f[0], x)
            f.flush()
    assert float(fits.getdata(p)[0, 0]) == 1000.0
    assert "BZERO" not in fits.getheader(p)


def test_qc_in_place_writer_uses_scale_safe_helper() -> None:
    src = (Path(__file__).resolve().parents[2] / "src_py" / "pipeline_calibrate.py").read_text(encoding="utf-8")
    assert "hdul[0].data = _as_fits_float32_image(data)" not in src
    assert "set_float_image_data_in_place(hdul[0]" in src


def test_sat_diag_reads_non_calibrated_lights(tmp_path: Path) -> None:
    _write_lights(tmp_path / "non_calibrated" / "lights" / "NoFilter_120_2", 6, n_sat=4)
    assert len(sample_light_paths(tmp_path)) == 6
    ctx = run_sat_diag(tmp_path)
    assert ctx.sat_source == "DERIVED" and ctx.measured_sat_adu() is not None
    assert ctx.sat_adu == pytest.approx(SAT, abs=60)
    assert ctx.pileup is not None and ctx.pileup.method == "star_peak_plateau"


def test_float_plateau_follows_the_data_units(tmp_path: Path) -> None:
    paths = _write_lights(tmp_path / "a", 6, n_sat=4, offset=32768.0)
    r = derive_ceiling_from_paths(paths)
    assert r.pileup_detected and r.v_ceiling == pytest.approx(SAT + 32768.0, abs=60)


def test_float_without_saturated_stars_is_unverified(tmp_path: Path) -> None:
    _write_lights(tmp_path / "non_calibrated" / "lights" / "g", 6, n_sat=0)
    ctx = run_sat_diag(tmp_path)
    assert ctx.sat_source == "MAX_OBSERVED" and ctx.sat_adu is not None
    assert not ctx.may_exclude_saturation() and ctx.measured_sat_adu() is None


def test_measured_level_wins_and_no_guessed_clip() -> None:
    # pixels near 65535 in data with an offset are not a clip any more
    arr = np.full((50, 50), 34000.0)
    arr[10:20, 10:20] = 65535.2
    arr[0, 0] = 98000.0
    assert _detect_empirical_clip_level_adu(arr) is None
    lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=65535.0, saturate_fraction=LINEARITY_FRAC, sky_median_adu=34000.0,
        frame_max_adu=98000.0, empirical_clip_adu=65535.0, measured_sat_adu=98278.0,
    )
    assert lim == pytest.approx(98278.0 * LINEARITY_FRAC)


def test_aperture_peak_test_from_sat_diag(tmp_path: Path) -> None:
    draft = tmp_path / "Drafts" / "draft_000999"
    ps = draft / "platesolve" / "NoFilter_120_2"
    ps.mkdir(parents=True)
    (draft / "draft_manifest.json").write_text("{}", encoding="ascii")
    assert draft_peak_test_adu(ps) == inv_sat_limit_peak_test_adu()  # no sat_diag.json
    _write_lights(draft / "non_calibrated" / "lights" / "NoFilter_120_2", 6, n_sat=4)
    write_sat_diag_json(run_sat_diag(draft), draft / "sat_diag.json")
    pt, src = draft_peak_test_adu(ps)
    assert pt == pytest.approx(SAT * LINEARITY_FRAC, abs=60) and "SAT-DIAG" in src


def test_old_sat_diag_json_with_refused_keys_loads(tmp_path: Path) -> None:
    import json

    from sat_diag import load_sat_diag_json

    p = tmp_path / "sat_diag.json"
    p.write_text(json.dumps({
        "sat_adu": 65535.0, "lin_adu": 58981.5, "sat_source": "DERIVED", "lin_source": "LINEARITY_FRAC",
        "pileup": {"pileup_detected": True, "v_ceiling": 65535.0, "n_at_ceiling": 2444, "n_at_shoulder": 0,
                   "max_pixel": 65535.0, "frames_sampled": 30, "bitpix_ceiling": 65535.0,
                   "refused": False, "refuse_reason": None},
    }), encoding="ascii")
    ctx = load_sat_diag_json(p)
    assert ctx is not None and ctx.sat_adu == 65535.0 and ctx.pileup.v_ceiling == 65535.0
