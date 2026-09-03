"""CONSOLIDATE-01E5: MP spawn smoke for the calibrate batch worker.

Imports the worker names from the pipeline facade so the test is unchanged
across C1 (signature fix) and C2 (pure move). After C2, pickle follows
__module__ into pipeline_calibrate in the spawn child. Note for E-final
retarget: this file still imports from pipeline on purpose.

Exercises the production initargs SHAPE (3-tuple) and one
_calibrate_batch_process_one item on a tiny synthetic light. No masters:
the disk path still writes dst and returns ok=True with VY_CFLAG=P
(see _calibration_flags: no dark, no flat, not an explicit passthrough
flag -> "P").
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits

from pipeline import _calibrate_batch_process_one, _init_calibrate_batch_worker


def _write_tiny_light(path: Path) -> None:
    hdr = fits.Header()
    hdr["FILTER"] = "NoFilter"
    hdr["EXPTIME"] = 1.0
    hdr["XBINNING"] = 1
    hdr["YBINNING"] = 1
    data = np.full((16, 16), 100.0, dtype=np.float32)
    fits.writeto(path, data, header=hdr, overwrite=True)


def test_calibrate_batch_mp_spawn_passthrough_roundtrip(tmp_path: Path) -> None:
    import multiprocessing

    src = tmp_path / "Light_001.fits"
    dst = tmp_path / "cal" / "Light_001.fits"
    _write_tiny_light(src)
    item = (
        str(src.resolve()),
        str(dst.resolve()),
        None,
        {},
        None,
    )
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        initializer=_init_calibrate_batch_worker,
        initargs=(None, 1, None),
    ) as ex:
        result = ex.submit(_calibrate_batch_process_one, item).result(timeout=25)

    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert result.get("error") is None
    assert result.get("src") == str(src.resolve())
    assert result.get("dst") == str(dst.resolve())
    assert Path(result["dst"]).is_file()
    assert str(result.get("vy_cflag") or "P").upper() == "P"
    with fits.open(result["dst"], memmap=False) as hdul:
        assert hdul[0].data.shape == (16, 16)
        assert bool(hdul[0].header.get("VYVARCAL")) is True
