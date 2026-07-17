#!/usr/bin/env python3
"""Maintenance: verify and correct OBS_DRAFT.ID_EQUIPMENTS from FITS header evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from astropy.io import fits  # noqa: E402

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402


def _parse_sensorsize(s: str | None) -> tuple[int | None, int | None]:
    if not s:
        return None, None
    txt = str(s).strip().lower().replace("x", "*")
    if "*" not in txt:
        return None, None
    a, b = txt.split("*", 1)
    try:
        return int(float(a.strip())), int(float(b.strip()))
    except ValueError:
        return None, None


def _sample_fits_headers(draft_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    lights_root = draft_dir / "non_calibrated" / "lights"
    if not lights_root.is_dir():
        return samples
    for setup_dir in sorted(lights_root.iterdir()):
        if not setup_dir.is_dir():
            continue
        fits_files = sorted(setup_dir.glob("*.fits"))
        if not fits_files:
            continue
        path = fits_files[0]
        with fits.open(path, memmap=False) as hdul:
            h = hdul[0].header
        samples.append(
            {
                "setup": setup_dir.name,
                "path": str(path),
                "INSTRUME": str(h.get("INSTRUME", "")).strip() or None,
                "NAXIS1": int(h["NAXIS1"]) if "NAXIS1" in h else None,
                "NAXIS2": int(h["NAXIS2"]) if "NAXIS2" in h else None,
                "XBINNING": int(h["XBINNING"]) if "XBINNING" in h else None,
                "YBINNING": int(h["YBINNING"]) if "YBINNING" in h else None,
                "GAIN": float(h["GAIN"]) if "GAIN" in h else None,
                "IMAGETYP": str(h.get("IMAGETYP", "")).strip() or None,
                "NCOMBINE": h.get("NCOMBINE"),
                "EXPTIME": float(h["EXPTIME"]) if "EXPTIME" in h else None,
            }
        )
    return samples


def _score_equipment_match(
    eq_row: dict[str, Any],
    sample: dict[str, Any],
) -> tuple[int, list[str]]:
    score = 0
    notes: list[str] = []
    instr = (sample.get("INSTRUME") or "").upper()
    cam = str(eq_row.get("CAMERANAME") or "").upper()
    if instr and cam and instr == cam:
        score += 100
        notes.append(f"INSTRUME matches CAMERANAME={eq_row.get('CAMERANAME')}")
    n1 = sample.get("NAXIS1")
    n2 = sample.get("NAXIS2")
    xb = sample.get("XBINNING") or 1
    yb = sample.get("YBINNING") or 1
    sw, sh = _parse_sensorsize(eq_row.get("SENSORSIZE"))
    if n1 and n2 and sw and sh and int(xb) == int(yb):
        if int(n1) * int(xb) == int(sw) and int(n2) * int(yb) == int(sh):
            score += 50
            notes.append(f"geometry {n1}x{n2} @ bin{xb} -> {sw}x{sh} matches SENSORSIZE")
    gain_hdr = sample.get("GAIN")
    gain_db = eq_row.get("GAIN_ADU")
    if gain_hdr is not None and gain_db is not None:
        exp = float(gain_db) * float(xb) ** 2
        if abs(float(gain_hdr) - exp) < 0.05 * max(exp, 1e-9):
            score += 10
            notes.append(f"GAIN {gain_hdr} matches bin-scaled DB {exp:.4f}")
    return score, notes


def verify_draft_equipment(
    draft_id: int,
    *,
    cfg: AppConfig,
) -> dict[str, Any]:
    db = VyvarDatabase(cfg.database_path)
    draft_row = db.conn.execute(
        "SELECT ID, ID_EQUIPMENTS, ID_TELESCOPE FROM OBS_DRAFT WHERE ID = ?;",
        (int(draft_id),),
    ).fetchone()
    if draft_row is None:
        return {"draft_id": draft_id, "error": "OBS_DRAFT not found"}
    current_eq = int(draft_row["ID_EQUIPMENTS"]) if draft_row["ID_EQUIPMENTS"] is not None else None
    eq_rows = [
        dict(r)
        for r in db.conn.execute(
            "SELECT ID, CAMERANAME, SENSORTYPE, SENSORSIZE, PIXELSIZE, GAIN_ADU, READNOISE_E FROM EQUIPMENTS;"
        ).fetchall()
    ]
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"
    samples = _sample_fits_headers(draft_dir)
    if not samples:
        return {
            "draft_id": draft_id,
            "current_equipment_id": current_eq,
            "error": "no FITS samples under non_calibrated/lights",
        }

    per_eq: list[dict[str, Any]] = []
    for eq in eq_rows:
        total = 0
        all_notes: list[str] = []
        for smp in samples:
            sc, notes = _score_equipment_match(eq, smp)
            total += sc
            all_notes.extend(notes)
        per_eq.append(
            {
                "equipment_id": int(eq["ID"]),
                "cameraname": eq.get("CAMERANAME"),
                "sensortype": eq.get("SENSORTYPE"),
                "score": total,
                "notes": all_notes,
            }
        )
    per_eq.sort(key=lambda x: int(x["score"]), reverse=True)
    best = per_eq[0] if per_eq else None
    second = per_eq[1] if len(per_eq) > 1 else None
    ambiguous = False
    verdict = "proceed"
    if best is None or int(best["score"]) < 50:
        verdict = "stop_ambiguous"
        ambiguous = True
    elif second is not None and int(second["score"]) >= int(best["score"]) - 10:
        verdict = "stop_ambiguous"
        ambiguous = True

    verified_id = int(best["equipment_id"]) if best and not ambiguous else None
    return {
        "draft_id": draft_id,
        "current_equipment_id": current_eq,
        "verified_equipment_id": verified_id,
        "verdict": verdict,
        "ambiguous": ambiguous,
        "header_samples": samples,
        "equipment_scores": per_eq,
        "gain_anomaly_note": (
            "Header GAIN may match a different equipment row than INSTRUME/geometry; "
            "do not override INSTRUME+geometry match on gain alone."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=int, required=True, help="OBS_DRAFT.ID to verify/fix")
    ap.add_argument("--apply", action="store_true", help="Write DB update when verified != current")
    ap.add_argument("--json", action="store_true", help="Print verification JSON")
    args = ap.parse_args()

    cfg = AppConfig()
    report = verify_draft_equipment(args.draft, cfg=cfg)
    cur = report.get("current_equipment_id")
    verified = report.get("verified_equipment_id")
    print(
        f"draft_{args.draft:06d} current_equipment_id={cur} "
        f"verified={verified} verdict={report.get('verdict')}"
    )
    if report.get("header_samples"):
        smp = report["header_samples"][0]
        print(
            f"  sample INSTRUME={smp.get('INSTRUME')} NAXIS={smp.get('NAXIS1')}x{smp.get('NAXIS2')} "
            f"bin={smp.get('XBINNING')} GAIN={smp.get('GAIN')}"
        )
    for row in report.get("equipment_scores", [])[:3]:
        print(f"  eq{row['equipment_id']} {row['cameraname']} score={row['score']}")

    if args.json:
        print(json.dumps(report, indent=2, default=str))

    if report.get("verdict") != "proceed" or verified is None:
        print("STOP: verification ambiguous or failed - no DB change.")
        return

    if int(verified) == int(cur):
        print("No change needed (already correct).")
        return

    if args.apply:
        db = VyvarDatabase(cfg.database_path)
        db.conn.execute(
            "UPDATE OBS_DRAFT SET ID_EQUIPMENTS = ? WHERE ID = ?;",
            (int(verified), int(args.draft)),
        )
        db.conn.commit()
        after = db.conn.execute(
            "SELECT ID_EQUIPMENTS FROM OBS_DRAFT WHERE ID = ?;", (int(args.draft),)
        ).fetchone()
        print(f"Applied: ID_EQUIPMENTS {cur} -> {after['ID_EQUIPMENTS']}")
    else:
        print(f"Would set ID_EQUIPMENTS={verified} (dry-run; pass --apply)")


if __name__ == "__main__":
    main()
