# -*- coding: ascii -*-
"""ANCHOR-HASH-01: content hash drops provenance headers; other bytes stay."""

from __future__ import annotations

from pathlib import Path

from tests.photometry_sha import (
    PHOTOMETRY_SHA_PROVENANCE_HEADER_KEYS,
    compute_photometry_sha,
    photometry_file_content_bytes,
)


def test_provenance_exclusion_keys_are_the_frozen_set() -> None:
    assert PHOTOMETRY_SHA_PROVENANCE_HEADER_KEYS == {
        "git_hash",
        "git_dirty",
        "files",
        "generated",
        "timestamp",
        "vyvar_version",
    }


def test_content_hash_ignores_git_hash_and_git_dirty(tmp_path: Path) -> None:
    a = tmp_path / "lightcurve_t_psf.csv"
    b = tmp_path / "other.csv"
    header_a = (
        b"# epsf_n_stars=12\n"
        b"# git_hash=aaaaaaaa\n"
        b"# git_dirty=True\n"
        b"x,y\n"
        b"1,2\n"
    )
    header_b = (
        b"# epsf_n_stars=12\n"
        b"# git_hash=bbbbbbbb\n"
        b"# git_dirty=False\n"
        b"x,y\n"
        b"1,2\n"
    )
    a.write_bytes(header_a)
    b.write_bytes(header_b)
    assert photometry_file_content_bytes(a) == photometry_file_content_bytes(b)
    assert photometry_file_content_bytes(a, strip_provenance=False) != header_b


def test_quoted_header_and_non_excluded_keys_stay_in_hash(tmp_path: Path) -> None:
    p = tmp_path / "lightcurve_t.csv"
    raw = b'# epsf_build_timestamp=2026-08-26\n# product="flux,adu"\nx,y\n1,2\n'
    p.write_bytes(raw)
    assert photometry_file_content_bytes(p) == raw


def test_tree_hash_matches_when_only_git_hash_differs(tmp_path: Path) -> None:
    for name, gh in (("snap", "aaa"), ("run", "bbb")):
        d = tmp_path / name / "platesolve" / "S" / "photometry" / "lightcurves"
        d.mkdir(parents=True)
        (d / "lightcurve_1_psf.csv").write_bytes(
            f"# git_hash={gh}\n# git_dirty=True\n# epsf_n_stars=1\na,b\n1,2\n".encode("ascii")
        )
        (d / "lightcurve_1.csv").write_bytes(b"a,b\n1,2\n")
    h1, n1 = compute_photometry_sha(tmp_path / "snap")
    h2, n2 = compute_photometry_sha(tmp_path / "run")
    assert n1 == n2 == 2
    assert h1 == h2
    raw1, _ = compute_photometry_sha(tmp_path / "snap", strip_provenance=False)
    raw2, _ = compute_photometry_sha(tmp_path / "run", strip_provenance=False)
    assert raw1 != raw2


def test_sha_split_core_aperture_excludes_psf(tmp_path: Path) -> None:
    from tests.photometry_sha import compute_photometry_sha_split

    phot = tmp_path / "platesolve" / "S" / "photometry"
    lc = phot / "lightcurves"
    lc.mkdir(parents=True)
    (lc / "lightcurve_1.csv").write_bytes(b"a\n1\n")
    (lc / "lightcurve_1_psf.csv").write_bytes(b"a\n9\n")
    (lc / "comp_quality_1.json").write_bytes(b"{}\n")
    (phot / "comparison_stars_per_target.csv").write_bytes(b"x\n1\n")
    ap, n_ap = compute_photometry_sha_split(tmp_path, "core_aperture")
    psf, n_psf = compute_photometry_sha_split(tmp_path, "core_psf")
    ext, n_ext = compute_photometry_sha_split(tmp_path, "ext_aperture")
    assert n_ap == 1
    assert n_psf == 1
    assert n_ext == 3
    assert ap != psf
