"""Resolve ``draft/.../platesolve`` artifacts for Streamlit (per-setup vs legacy root)."""

from __future__ import annotations

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
