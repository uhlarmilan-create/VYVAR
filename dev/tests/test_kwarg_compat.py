"""PP-KWARG-01: static kwarg signature compatibility gate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "tools"))

from kwarg_compat_scan import scan_source_text, scan_src_py  # noqa: E402


def test_pp_kwarg01_production_scope_clean() -> None:
    hits = scan_src_py()
    assert hits == [], "PP-KWARG-01 kwarg mismatches:\n" + "\n".join(
        f"{v.module}:{v.line} {v.callee}() {v.bad_kw}" for v in hits
    )


def test_pp_kwarg01_fire_proof_detects_splat_mismatch() -> None:
    fixture = '''
from pipeline import qc_enrich_calibrated_lights_in_place

def _run(root, cfg):
    _pp_kw = dict(
        reject_fwhm_px=None,
        use_gpu_if_available=False,
        app_config=cfg,
    )
    qc_enrich_calibrated_lights_in_place(
        calibrated_root=root,
        **_pp_kw,
    )
'''
    hits = scan_source_text(fixture, module="fixture.py")
    bad = {v.bad_kw for v in hits}
    assert "use_gpu_if_available" in bad
