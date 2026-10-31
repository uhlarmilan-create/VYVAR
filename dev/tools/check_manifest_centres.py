#!/usr/bin/env python3
"""Read-only check of the manifest field centres of processed fields (G14-03; S6-FIX CENTRE-PER-SETUP,
Milan 2026-10-31).

Every setup of a field is its own instrument, so the manifest keeps one centre per setup
(``center_by_setup``). For every draft under the given ``Archive/Drafts`` root(s) this compares the
centre of each setup with the WCS centre of THAT setup's ``platesolve/<setup>/MASTERSTAR.fits`` and lists
the mismatches larger than 10 % of the field radius (half the frame diagonal), distance in arcmin. A setup
without a stored centre is ``unset`` (drafts processed before step 6; the next night re-run writes it). A
legacy field-level ``center`` (written by the GUI before step 6) is reported as ``legacy center ignored``:
nothing reads it any more. Nothing is written.

    python dev/tools/check_manifest_centres.py C:\\ASTRO\\VYVAR\\Archive\\Drafts

Exit code 0 = no mismatch, 1 = at least one mismatch (or a 0/0 centre).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

#: A manifest centre further than this fraction of the MASTERSTAR field radius is reported.
MISMATCH_FRACTION_OF_RADIUS = 0.10


def check_draft(draft_dir: Path) -> list[dict]:
    """One row per MASTERSTAR of the draft: setup, manifest centre, WCS centre, distance, status."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    from manifest_centre import masterstar_files, masterstar_wcs_centre

    man = json.loads((draft_dir / "draft_manifest.json").read_text(encoding="utf-8"))
    centres = man.get("center_by_setup") if isinstance(man.get("center_by_setup"), dict) else {}
    rows: list[dict] = []
    legacy = man.get("center") if isinstance(man.get("center"), dict) else {}
    if legacy.get("ra_deg") is not None:
        rows.append({"draft": draft_dir.name, "setup": "-", "manifest_ra": legacy.get("ra_deg"),
                     "manifest_de": legacy.get("de_deg"), "status": "legacy center ignored"})
    for ms in masterstar_files(draft_dir):
        c = centres.get(ms.parent.name) if isinstance(centres.get(ms.parent.name), dict) else {}
        ra, de = c.get("ra_deg"), c.get("de_deg")
        row = {"draft": draft_dir.name, "setup": ms.parent.name, "manifest_ra": ra, "manifest_de": de}
        try:
            wc = masterstar_wcs_centre(ms)
        except (OSError, ValueError) as exc:
            rows.append({**row, "status": f"MASTERSTAR unreadable: {exc}"})
            continue
        if wc is None:
            rows.append({**row, "status": "MASTERSTAR without solved WCS"})
            continue
        row.update(wcs_ra=wc[0], wcs_de=wc[1], radius_arcmin=60.0 * wc[2])
        if ra is None or de is None:
            rows.append({**row, "status": "unset"})
            continue
        if abs(float(ra)) < 1e-9 and abs(float(de)) < 1e-9:
            rows.append({**row, "status": "MISMATCH (0/0)"})
            continue
        sep = SkyCoord(float(ra) * u.deg, float(de) * u.deg).separation(SkyCoord(wc[0] * u.deg, wc[1] * u.deg))
        d_arcmin = float(sep.arcmin)
        limit = MISMATCH_FRACTION_OF_RADIUS * 60.0 * wc[2]
        row["distance_arcmin"] = d_arcmin
        rows.append({**row, "status": "MISMATCH" if d_arcmin > limit else "ok"})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drafts_root", type=Path, nargs="+", help="Archive/Drafts folder(s)")
    ap.add_argument("--all", action="store_true", help="print every MASTERSTAR, not only the mismatches")
    args = ap.parse_args(argv)
    n_bad = n_checked = 0
    for root in args.drafts_root:
        for man in sorted(Path(root).glob("*/draft_manifest.json")):
            for r in check_draft(man.parent):
                n_checked += 1
                bad = str(r["status"]).startswith("MISMATCH")
                n_bad += int(bad)
                if bad or args.all:
                    dist = f"{r['distance_arcmin']:.2f} arcmin" if "distance_arcmin" in r else "-"
                    rad = f"{r['radius_arcmin']:.1f} arcmin" if "radius_arcmin" in r else "-"
                    print(f"{r['draft']}\t{r['setup']}\t{r['status']}\tdistance {dist}\tfield radius {rad}\t"
                          f"manifest ({r['manifest_ra']}, {r['manifest_de']})\t"
                          f"WCS ({r.get('wcs_ra', '-')}, {r.get('wcs_de', '-')})")
    print(f"{n_checked} MASTERSTAR(s) checked, {n_bad} mismatch(es)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
