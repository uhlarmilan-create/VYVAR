"""Phase-1 open LOW items (code review 2): G04-38, G06-31, G09-35, G10-36, G10-38, G11-39."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_pass2_catalog_equals_public_finder(seed: int) -> None:
    """G04-38: the cached pass-2 finder gives the public DAOStarFinder catalogue and is not mutated."""
    from photutils.detection import DAOStarFinder

    import masterstar_gaia_accounting as mga
    from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:41, 0:41]
    img = rng.normal(100.0, 5.0, (41, 41))
    for _ in range(4):
        x0, y0, a = rng.uniform(5, 35), rng.uniform(5, 35), rng.uniform(50, 800)
        img += a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.6**2))
    img -= np.median(img)
    for thr in (8.0, 20.0, 60.0):
        got = mga._pass2_catalog(img, 3.7, thr)
        ref = DAOStarFinder(fwhm=3.7, threshold=thr, n_brightest=None, **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER)(img)
        if ref is None:
            assert got is None or len(got.xcentroid) == 0
            continue
        assert np.allclose(np.sort(np.asarray(got.xcentroid)), np.sort(np.asarray(ref["xcentroid"])))
        assert np.allclose(np.sort(np.asarray(got.flux)), np.sort(np.asarray(ref["flux"])))
    assert mga._PASS2_FINDERS[3.7].threshold == 1.0


def test_net_call_timeout_and_hold() -> None:
    """G10-36: one timeout / unavailable registry for every web service."""
    import threading

    import net_call

    ev = threading.Event()
    with pytest.raises(TimeoutError):
        net_call.call_with_timeout(lambda: ev.wait(5), 1.0, "slow")
    ev.set()
    assert net_call.call_with_timeout(lambda: 7, 1.0, "fast") == 7
    assert net_call.unavailable_reason("svc") is None
    net_call.mark_unavailable("svc", RuntimeError("down"))
    assert net_call.unavailable_reason("svc") == "down"
    net_call.clear_unavailable()
    assert net_call.unavailable_reason("svc") is None


def test_no_private_network_state_outside_net_call() -> None:
    """G10-36 / G10-38: no own unavailable dict or timeout helper, no global Gaia.ROW_LIMIT."""
    bad = []
    for p in sorted((ROOT / "src_py").glob("*.py")):
        if p.name == "net_call.py":
            continue
        t = p.read_text(encoding="utf-8")
        for s in (r"\b_UNAVAILABLE\b", r"\b_call_with_timeout\b", r"\bROW_LIMIT\b"):
            if re.search(s, t):
                bad.append(f"{p.name}: {s}")
    assert bad == []


def test_provenance_runs_git_status_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """G11-39: one ``git status`` per stamp; the classification needs only the paths."""
    import photometry_provenance as pp

    calls: list[list[str]] = []

    def _fake(cmd: list[str], **_k: object) -> str:
        calls.append(cmd)
        return "abc123\n" if cmd[1] == "rev-parse" else " M src_py/x.py\n?? dev/tmp.md\n"

    monkeypatch.setattr(pp.subprocess, "check_output", _fake)
    block = pp._build_pipeline_provenance_block(type("C", (), {"to_dict": lambda self: {}})(), entry_point="t")
    assert sum(1 for c in calls if c[1] == "status") == 1
    assert block["git_dirty_code_files"] == ["src_py/x.py"]
    assert block["git_dirty_scratch_files"] == ["dev/tmp.md"]
    assert not hasattr(pp, "_GIT_PROVENANCE_WARNED")


def test_state_builder_does_not_write_active_targets() -> None:
    """G06-31: the shared-state builder only loads; the driver writes active_targets.csv."""
    src = (ROOT / "src_py" / "aperture_photometry_state.py").read_text(encoding="utf-8")
    assert "to_csv(active_targets_csv" not in src
    assert "at_df.to_csv(active_targets_csv" in (ROOT / "src_py" / "aperture_photometry.py").read_text(encoding="utf-8")


def test_epsf_export_uses_public_helpers() -> None:
    """G09-35: no private imports from report modules, oversampling from the manifest."""
    import ast

    t = ast.parse((ROOT / "src_py" / "epsf_export.py").read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(t) if isinstance(n, ast.ImportFrom) for a in n.names]
    assert [n for n in names if n.startswith("_")] == []
    assert "OVERSAMPLING" not in names
