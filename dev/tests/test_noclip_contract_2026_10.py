"""INV-NOCLIP-01 contract (code review 2, T20 / E6): no sigma clipping anywhere, also not hidden in library defaults.

Static scan of src_py: library estimators whose default clips (photutils MedianBackground / MMMBackground /
SExtractorBackground / ModeEstimatorBackground / Background2D / LocalBackground / EPSFBuilder / SigmaClip,
astropy sigma_clip / sigma_clipped_stats) may only be built with an explicit ``sigma_clip=None``; estimators
whose default does not clip (photutils ApertureStats) must not be given a clip. ``plain_mean_med_std`` accepts
no clip arguments.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CLIP_BY_DEFAULT = {
    "MedianBackground", "MeanBackground", "MMMBackground", "SExtractorBackground", "ModeEstimatorBackground",
    "BiweightLocationBackground", "Background2D", "LocalBackground", "EPSFBuilder", "StdBackgroundRMS",
    "MADStdBackgroundRMS", "BiweightScaleBackgroundRMS",
}
FORBIDDEN = {"SigmaClip", "sigma_clip", "sigma_clipped_stats"}
NO_CLIP_DEFAULT = {"ApertureStats"}


def _calls():
    for p in sorted(SRC.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="ascii"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
                if name:
                    yield p.name, node.lineno, name, node


def _sigma_clip_kw(node):
    for kw in node.keywords:
        if kw.arg == "sigma_clip":
            return kw.value
    return "absent"


def test_no_clipping_estimator_is_built():
    bad = []
    for fn, line, name, node in _calls():
        if name in FORBIDDEN:
            bad.append(f"{fn}:{line} {name}")
        elif name in CLIP_BY_DEFAULT:
            v = _sigma_clip_kw(node)
            if not (isinstance(v, ast.Constant) and v.value is None):
                bad.append(f"{fn}:{line} {name} without sigma_clip=None")
        elif name in NO_CLIP_DEFAULT:
            v = _sigma_clip_kw(node)
            if not (v == "absent" or (isinstance(v, ast.Constant) and v.value is None)):
                bad.append(f"{fn}:{line} {name} with a clip")
    assert not bad, bad


def test_library_defaults_still_as_assumed():
    from photutils.aperture import ApertureStats

    assert inspect.signature(ApertureStats.__init__).parameters["sigma_clip"].default is None


def test_plain_statistics_accept_no_clip_arguments():
    from plain_stats import plain_mean_med_std

    assert plain_mean_med_std([1.0, 2.0, 100.0])[1] == 2.0
    with pytest.raises(TypeError):
        plain_mean_med_std([1.0, 2.0], sigma=3.0)
