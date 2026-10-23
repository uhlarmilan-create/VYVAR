"""Resolve ``draft/.../platesolve`` artifacts for Streamlit (per-setup vs legacy root)."""

from __future__ import annotations

import re
from pathlib import Path


def masterstars_csv_in_dir(d: Path) -> Path | None:
    for name in ("masterstars_full_match.csv", "masterstars.csv"):
        p = d / name
        if p.is_file():
            return p
    return None


def list_platesolve_setup_dirs(ps_root: Path) -> list[Path]:
    """Subdirectories that look like a filter/setup (index and/or masterstar bundle)."""
    if not ps_root.is_dir():
        return []
    out: list[Path] = []
    for sub in sorted(ps_root.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "per_frame_catalog_index.csv").is_file():
            out.append(sub)
            continue
        if (sub / "MASTERSTAR.fits").is_file() and masterstars_csv_in_dir(sub):
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
    for sub in list_platesolve_setup_dirs(ps_root):
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


def parse_draft_id_from_text(text: str) -> int | None:
    """Extract ``draft_000229`` / ``draft-229`` style id from free text or a path."""
    s = (text or "").strip().strip('"').strip("'")
    if not s:
        return None
    if s.isdigit():
        return int(s)
    m = re.search(r"draft[_-]?(\d{1,8})(?:\D|$)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def resolve_draft_directory(
    text: str,
    *,
    archive_root: Path,
) -> tuple[Path | None, int | None, str]:
    """Resolve a user-supplied draft folder or id.

    Returns ``(draft_dir, draft_id, error_message)``. ``error_message`` is empty on success.
    ``draft_dir`` is the directory that should contain a ``platesolve/`` subfolder.
    """
    s = (text or "").strip().strip('"').strip("'")
    if not s:
        return None, None, ""

    archive_root = Path(archive_root)
    draft_id = parse_draft_id_from_text(s)

    p = Path(s).expanduser()
    if p.is_dir():
        ps = p / "platesolve"
        if ps.is_dir():
            if draft_id is None:
                m = re.search(r"draft[_-]?(\d{1,8})", p.name, flags=re.IGNORECASE)
                draft_id = int(m.group(1)) if m else None
            return p.resolve(), draft_id, ""
        return None, draft_id, f"No ``platesolve/`` under the folder: {p}"

    if draft_id is not None:
        cand = (archive_root / "Drafts" / f"draft_{int(draft_id):06d}").resolve()
        if cand.is_dir() and (cand / "platesolve").is_dir():
            return cand, draft_id, ""
        return None, draft_id, f"Draft {draft_id} does not exist in archive: {cand}"

    return None, None, "Enter the full path to the ``draft_XXXXXX`` folder or the draft number."


def setup_label(name: str) -> str:
    """``NoFilter_60_2`` -> ``NoFilter (dot) 60 s (dot) bin 2``; other folder names unchanged."""
    m = re.fullmatch(r"(.+)_(\d+(?:\.\d+)?)_(\d+)", str(name))
    return f"{m.group(1)} \u00b7 {m.group(2)} s \u00b7 bin {m.group(3)}" if m else str(name)


def find_photometry_setups(cfg, draft_id: int | None, *, draft_dir_override: Path | None = None) -> dict[str, dict]:
    """All observation setups of a draft (``platesolve/<setup>`` with a per-frame catalog index),
    keyed by setup name, each with its standard paths."""
    from utils import resolve_draft_dir_path  # noqa: PLC0415

    if draft_id is None and draft_dir_override is None:
        return {}
    draft_dir = resolve_draft_dir_path(draft_dir_override, draft_id, cfg.archive_root)
    if draft_dir is None:
        return {}
    ps_dir = draft_dir / "platesolve"
    aligned_root = draft_dir / "detrended_aligned" / "lights"
    if not ps_dir.exists():
        return {}
    result: dict[str, dict] = {}
    for subdir in sorted(ps_dir.iterdir()):
        if not subdir.is_dir() or not (subdir / "per_frame_catalog_index.csv").exists():
            continue
        setup_name = subdir.name
        per_frame_dir = (aligned_root / setup_name) if (aligned_root / setup_name).exists() else None
        photometry_dir = subdir / "photometry"
        result[str(setup_name)] = {
            "setup_name": subdir,
            "masterstar_fits": subdir / "MASTERSTAR.fits",
            "active_targets_csv": photometry_dir / "active_targets.csv",
            "comparison_stars_csv": photometry_dir / "comparison_stars_per_target.csv",
            "per_frame_csv_dir": per_frame_dir,
            "detrended_aligned_dir": per_frame_dir,
            "output_dir": photometry_dir,
            "photometry_dir": photometry_dir,
            "obs_group_dir": subdir,
        }
    return result


def masterstar_fwhm_px(masterstar_fits: Path | None) -> float:
    """FWHM (px) stamped on MASTERSTAR.fits (VY_FWHM_GAUSS / VY_FWHM); 3.7 when absent."""
    if masterstar_fits is None or not Path(masterstar_fits).is_file():
        return 3.7
    from astropy.io import fits as astrofits  # noqa: PLC0415

    try:
        with astrofits.open(masterstar_fits, memmap=False) as hdul:
            for key in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN", "VY_FWHM"):
                v = hdul[0].header.get(key)
                if v is not None:
                    fv = float(v)
                    if 0.5 < fv < 30.0:
                        return round(fv, 3)
    except (OSError, ValueError, TypeError):
        pass
    return 3.7
