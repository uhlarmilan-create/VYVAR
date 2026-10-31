"""Resolve ``draft/.../platesolve`` artifacts for Streamlit (per-setup vs legacy root)."""

from __future__ import annotations

import re
from pathlib import Path


def masterstars_csv_in_dir(d: Path | str) -> Path | None:
    """Masterstars CSV of a platesolve setup dir (full_match preferred), or None. G13-28: the one resolver."""
    for name in ("masterstars_full_match.csv", "masterstars.csv"):
        p = Path(d) / name
        if p.is_file():
            return p
    return None


def setup_dirs(ps_root: Path, *, require_index: bool) -> list[Path]:
    """Setup (filter / obs group) subdirectories of ``platesolve/``, sorted by name (G03-34: the one rule).

    A setup has a per-frame catalog index; with ``require_index=False`` a MASTERSTAR bundle
    (MASTERSTAR.fits + masterstars CSV) without an index counts too (UI listing before photometry)."""
    ps_root = Path(ps_root)
    if not ps_root.is_dir():
        return []
    out: list[Path] = []
    for sub in sorted(ps_root.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "per_frame_catalog_index.csv").is_file():
            out.append(sub)
        elif not require_index and (sub / "MASTERSTAR.fits").is_file() and masterstars_csv_in_dir(sub):
            out.append(sub)
    return out


def legacy_platesolve_root_bundle(ps_root: Path) -> Path | None:
    """``platesolve/`` root with MASTERSTAR + catalog (pre per-setup layout)."""
    if not ps_root.is_dir():
        return None
    if (ps_root / "MASTERSTAR.fits").is_file() and masterstars_csv_in_dir(ps_root):
        return ps_root
    return None


def platesolve_bundle_dirs(ps_root: Path) -> list[Path]:
    """Ordered dirs that contain ``MASTERSTAR.fits`` and a masterstars CSV."""
    bundles: list[Path] = []
    for sub in setup_dirs(ps_root, require_index=False):
        if (sub / "MASTERSTAR.fits").is_file() and masterstars_csv_in_dir(sub):
            bundles.append(sub)
    leg = legacy_platesolve_root_bundle(ps_root)
    if leg is not None and not any(p.resolve() == leg.resolve() for p in bundles):
        bundles.insert(0, leg)
    return bundles


def default_bundle_dir(ps_root: Path, *, preferred_name: str | None = None) -> Path | None:
    """Pick one setup dir: explicit name, else first ``R_*``, else first bundle, else legacy root."""
    bundles = platesolve_bundle_dirs(ps_root)
    if not bundles:
        return None
    if preferred_name:
        for p in bundles:
            if p.name == preferred_name:
                return p
    r_pref = [p for p in bundles if p.name.upper().startswith("R_")]
    if r_pref:
        return r_pref[0]
    return bundles[0]


def cone_csv_path(setup_dir: Path) -> Path:
    return setup_dir / "field_catalog_cone.csv"


def resolve_draft_directory(
    text: str,
    *,
    archive_root: Path,
) -> tuple[Path | None, int | None, str]:
    """Resolve a processed field given as a folder path, a folder name under ``Archive/Drafts``
    (``MN_Boo_20260422_001``, legacy ``draft_000538``) or its number (PROCESSED-FIELD-NAME-2026-10).

    Returns ``(folder, id, error_message)``; ``error_message`` is empty on success. The folder
    must contain ``platesolve/``."""
    from draft_provenance import draft_dir_for_id  # noqa: PLC0415
    from processed_field import processed_field_id  # noqa: PLC0415

    s = (text or "").strip().strip('"').strip("'")
    if not s:
        return None, None, ""
    archive_root = Path(archive_root)
    if s.isdigit():
        cand = draft_dir_for_id(archive_root, int(s)).resolve()
    else:
        p = Path(s).expanduser()
        cand = p if p.is_dir() else (archive_root / "Drafts" / s)
    if not cand.is_dir():
        return None, None, f"Processed field not found: {s} (looked in {archive_root / 'Drafts'})"
    if not (cand / "platesolve").is_dir():
        return None, processed_field_id(cand), f"No ``platesolve/`` in the processed field folder: {cand}"
    return cand.resolve(), processed_field_id(cand), ""


def setup_label(name: str) -> str:
    """``NoFilter_60_2`` -> ``NoFilter (dot) 60 s (dot) bin 2``; other folder names unchanged."""
    m = re.fullmatch(r"(.+)_(\d+(?:\.\d+)?)_(\d+)", str(name))
    return f"{m.group(1)} \u00b7 {m.group(2)} s \u00b7 bin {m.group(3)}" if m else str(name)


def find_photometry_setups(cfg, draft_id: int | None, *, draft_dir_override: Path | None = None) -> dict[str, dict]:
    """All observation setups of a draft (``platesolve/<setup>`` with a per-frame catalog index),
    keyed by setup name, each with its standard paths (one key per path; G03-34)."""
    from draft_provenance import draft_dir_for_id  # noqa: PLC0415

    # CORE-NO-UI-2026-10: the given folder or the folder of draft_id - never GUI session state
    # (with session state the night run could take the setups of the GUI's last job).
    if draft_dir_override is not None:
        draft_dir = Path(draft_dir_override).expanduser().resolve()
    elif draft_id is not None:
        draft_dir = draft_dir_for_id(cfg.archive_root, int(draft_id)).resolve()
    else:
        return {}
    ps_dir = draft_dir / "platesolve"
    aligned_root = draft_dir / "detrended_aligned" / "lights"
    result: dict[str, dict] = {}
    for subdir in setup_dirs(ps_dir, require_index=True):
        setup_name = subdir.name
        per_frame_dir = (aligned_root / setup_name) if (aligned_root / setup_name).exists() else None
        photometry_dir = subdir / "photometry"
        result[str(setup_name)] = {
            "setup_name": str(setup_name),
            "masterstar_fits": subdir / "MASTERSTAR.fits",
            "active_targets_csv": photometry_dir / "active_targets.csv",
            "comparison_stars_csv": photometry_dir / "comparison_stars_per_target.csv",
            "per_frame_csv_dir": per_frame_dir,  # the aligned frames (also the detrended_aligned dir)
            "photometry_dir": photometry_dir,
            "obs_group_dir": subdir,
        }
    return result


def masterstar_fwhm_px(masterstar_fits: Path | None) -> float:
    """FWHM (px) stamped on MASTERSTAR.fits (VY_FWHM, the one reader of G03-29), rounded to 3 decimals.

    A missing file or VY_FWHM raises ValueError (was a silent 3.7 px)."""
    from astropy.io import fits as astrofits  # noqa: PLC0415

    from aperture_policy import fwhm_from_header_vy_fwhm  # noqa: PLC0415

    if masterstar_fits is None or not Path(masterstar_fits).is_file():
        raise ValueError(f"MASTERSTAR not found: {masterstar_fits}")
    fv = fwhm_from_header_vy_fwhm(astrofits.getheader(str(masterstar_fits)))
    if fv is None:
        raise ValueError(f"MASTERSTAR {masterstar_fits} has no valid VY_FWHM")
    return round(fv, 3)
