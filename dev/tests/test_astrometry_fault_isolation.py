"""Per-set astrometry fault isolation (multi-group plate-solve / MASTERSTAR)."""
from __future__ import annotations

from pathlib import Path

import pytest


def _make_two_group_archive(tmp_path: Path) -> Path:
    ap = tmp_path / "draft_test"
    for name in ("g_60_4", "r_60_4"):
        d = ap / "processed" / "lights" / name
        d.mkdir(parents=True)
        (d / f"proc_{name}.fits").write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    return ap


def _make_single_group_archive(tmp_path: Path) -> Path:
    ap = tmp_path / "draft_single"
    d = ap / "processed" / "lights" / "g_60_4"
    d.mkdir(parents=True)
    (d / "proc_g.fits").write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    return ap


def test_astrometry_multi_group_skips_failed_set(tmp_path: Path, monkeypatch):
    import pipeline as pl

    ap = _make_two_group_archive(tmp_path)

    def _fake_impl(*, job, **_kwargs):
        gkey = str(job.get("gkey") or "")
        setup = Path(gkey).name if gkey else "(root)"
        if setup == "r_60_4":
            raise RuntimeError("simulated plate-solve fail")
        return {
            "gkey": gkey,
            "aligned_frames": 10,
            "input_frames": 10,
            "masterstar_fits": f"/fake/{setup}/MASTERSTAR.fits",
            "solved": True,
        }

    monkeypatch.setattr(pl, "_astrometry_align_impl_body", _fake_impl)

    result = pl.astrometry_align_and_build_masterstar(archive_path=ap)

    assert int(result.get("aligned_frames") or 0) == 10
    skipped = result.get("skipped_subgroups") or []
    assert len(skipped) == 1
    assert skipped[0]["setup"] == "r_60_4"
    assert skipped[0]["solved"] is False
    assert "simulated plate-solve fail" in str(skipped[0]["skipped_reason"])
    assert "g_60_4" in str(result.get("masterstar_fits") or "")


def test_astrometry_multi_group_all_fail_raises(tmp_path: Path, monkeypatch):
    import pipeline as pl

    ap = _make_two_group_archive(tmp_path)

    def _always_fail(*, job, **_kwargs):
        gkey = str(job.get("gkey") or "")
        setup = Path(gkey).name if gkey else "(root)"
        raise RuntimeError(f"boom-{setup}")

    monkeypatch.setattr(pl, "_astrometry_align_impl_body", _always_fail)

    with pytest.raises(RuntimeError, match="žiadny set neprešiel"):
        pl.astrometry_align_and_build_masterstar(archive_path=ap)


def test_astrometry_single_group_failure_propagates(tmp_path: Path, monkeypatch):
    import pipeline as pl

    ap = _make_single_group_archive(tmp_path)

    def _fail(*, job, **_kwargs):
        raise RuntimeError("solo fail")

    monkeypatch.setattr(pl, "_astrometry_align_impl_body", _fail)

    with pytest.raises(RuntimeError, match="solo fail"):
        pl.astrometry_align_and_build_masterstar(archive_path=ap)
