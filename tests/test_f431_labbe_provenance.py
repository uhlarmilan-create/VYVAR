"""F-431: Labbe RNG seed + git_dirty file list provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from photometry_core import (
    _build_pipeline_provenance_block,
    _labbe_content_seed_from_header,
    measure_empty_aperture_sigma_bkg,
)


def test_labbe_seed_deterministic_same_content() -> None:
    rng_data = np.random.default_rng(0)
    data = rng_data.normal(1000.0, 20.0, size=(200, 200))
    xs = np.array([50.0, 100.0, 150.0])
    ys = np.array([60.0, 110.0, 140.0])
    hdr = fits.Header()
    hdr["DATE-OBS"] = "2026-04-23T19:49:27"
    hdr["FILENAME"] = "proc_BO_CVn_Light_008.fits"
    hdr["NAXIS1"] = 200
    hdr["NAXIS2"] = 200
    seed = _labbe_content_seed_from_header(hdr, r_ap=3.0)
    a, _, _ = measure_empty_aperture_sigma_bkg(
        data, xs, ys, 3.0, 5.0, 9.0, n_apertures=32, min_valid=8, seed=seed
    )
    b, _, _ = measure_empty_aperture_sigma_bkg(
        data, xs, ys, 3.0, 5.0, 9.0, n_apertures=32, min_valid=8, seed=seed
    )
    assert np.isfinite(a) and np.isfinite(b)
    assert a == b


def test_provenance_includes_dirty_files_when_dirty(tmp_path: Path, monkeypatch) -> None:
    # Force provenance resolver to report dirty via monkeypatch of subprocess? Simpler:
    # stub _resolve_git_provenance
    import photometry_core as pc

    monkeypatch.setattr(
        pc,
        "_resolve_git_provenance",
        lambda: ("deadbeef", True, [{"path": "foo.py", "content_sha256": "abc"}]),
    )

    class _Cfg:
        def to_dict(self):
            return {"sysrem_enabled": False}

    block = _build_pipeline_provenance_block(_Cfg(), entry_point="test")
    assert block["git_dirty"] is True
    assert block["git_dirty_files"][0]["path"] == "foo.py"
    assert block["labbe_rng_seed_policy"] == "content_frame_hash_v1"
