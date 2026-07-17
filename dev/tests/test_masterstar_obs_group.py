"""Tests for per-obs_group MASTERSTAR input root resolution (no cross-group leak)."""
from __future__ import annotations

from pathlib import Path

import pytest


def _make_multi_group_archive(tmp_path: Path) -> Path:
    ap = tmp_path / "draft_test"
    for name in ("Blue_60_2", "Green_60_2", "Red_60_2"):
        d = ap / "processed" / "lights" / name
        d.mkdir(parents=True)
        (d / f"proc_{name}.fits").write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    return ap


def test_resolve_masterstar_input_root_exact_setup_tmp(tmp_path: Path):
    from pipeline import resolve_masterstar_input_root

    ap = _make_multi_group_archive(tmp_path)
    green = resolve_masterstar_input_root(ap, setup_name="Green_60_2")
    assert green is not None
    assert green.name == "Green_60_2"
    assert green.parent.name == "lights"

    missing = resolve_masterstar_input_root(ap, setup_name="Nope_60_2")
    assert missing is None

    blue = resolve_masterstar_input_root(ap, setup_name="Blue_60_2")
    assert blue is not None
    assert blue.name == "Blue_60_2"
    assert green.resolve() != blue.resolve()


def test_resolve_masterstar_input_root_legacy_single_group_scan(tmp_path: Path):
    from pipeline import resolve_masterstar_input_root

    ap = tmp_path / "single"
    d = ap / "processed" / "lights" / "NoFilter_60_2"
    d.mkdir(parents=True)
    (d / "proc_a.fits").write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    hit = resolve_masterstar_input_root(ap, setup_name=None)
    assert hit is not None
    assert hit.name == "NoFilter_60_2"


def test_draft_is_multi_group_obs(tmp_path: Path):
    from pipeline import draft_is_multi_group_obs, draft_obs_group_count

    ap = _make_multi_group_archive(tmp_path)
    assert draft_obs_group_count(ap) == 3
    assert draft_is_multi_group_obs(ap) is True

    single = tmp_path / "one"
    d = single / "processed" / "lights" / "Blue_60_2"
    d.mkdir(parents=True)
    (d / "proc_x.fits").write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    assert draft_is_multi_group_obs(single) is False
