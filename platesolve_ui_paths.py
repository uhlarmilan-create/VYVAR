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
        return None, draft_id, f"Pod adresárom nie je ``platesolve/``: {p}"

    if draft_id is not None:
        cand = (archive_root / "Drafts" / f"draft_{int(draft_id):06d}").resolve()
        if cand.is_dir() and (cand / "platesolve").is_dir():
            return cand, draft_id, ""
        return None, draft_id, f"Draft {draft_id} neexistuje v archíve: {cand}"

    return None, None, "Zadaj celú cestu k priečinku ``draft_XXXXXX`` alebo číslo draftu."
