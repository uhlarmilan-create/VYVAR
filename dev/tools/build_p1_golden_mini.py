# -*- coding: ascii -*-
"""Build the INVARIANTS P1 golden mini-dataset from draft_000435.

Runbook (local):
  python dev/tools/build_p1_golden_mini.py
  set VYVAR_INVARIANTS_P1=1
  pytest dev/tests/test_invariants_p1_seed.py dev/tests/test_invariants_p1_golden.py -q

Design (see DECISIONS INVARIANTS-P1-GOLDEN-MINI):
  - Source: Archive/Drafts/draft_000435
  - 16 lights by even DATE-OBS stride (first included)
  - Photometry-ready layout: calibrated subset + detrended_aligned proc
    CSV/FITS for those frames + platesolve catalogs/MASTERSTAR from parent
  - Scope: in-draft Raw/darks/flats have no local masters (CalibrationLibrary);
    mini starts at photometry-ready stage matching session_baseline --full
  - Idempotent: wipe + rebuild; prints frame list + SHA256 per copied input
  - Writes p1_manifest.json next to the mini root
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src_py"))

from config import AppConfig  # noqa: E402

SOURCE_DRAFT = "draft_000435"
MINI_NAME = "draft_000435_p1mini"
SETUP = "NoFilter_60_2"
N_FRAMES = 16


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_into(mini_root: Path, src: Path, rel: str, inputs: list[dict]) -> Path:
    dst = mini_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = _sha256_file(dst)
    inputs.append({"rel": rel.replace("\\", "/"), "sha256": digest, "bytes": dst.stat().st_size})
    return dst


def select_stride_frames(
    cal_dir: Path,
    da_dir: Path,
    n: int = N_FRAMES,
) -> list[tuple[str, str]]:
    """Return [(DATE-OBS, calibrated_basename), ...] length n.

    Even stride over time-sorted frames that have BOTH a calibrated FITS and a
    matching detrended_aligned proc_*.fits (QC-rejected frames have no proc).
    First frame included.
    """
    from astropy.io import fits

    rows: list[tuple[str, str]] = []
    for f in sorted(cal_dir.glob("*.fits")):
        proc = da_dir / f"proc_{f.stem}.fits"
        if not proc.is_file():
            continue
        with fits.open(f, memmap=True) as hdul:
            dobs = str(hdul[0].header.get("DATE-OBS") or hdul[0].header.get("DATE") or "")
        rows.append((dobs, f.name))
    rows.sort(key=lambda t: t[0])
    if len(rows) < n:
        raise RuntimeError(
            f"need >= {n} calibrated+aligned lights, found {len(rows)} in {cal_dir}"
        )
    idxs = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    seen: set[int] = set()
    out: list[tuple[str, str]] = []
    for i in idxs:
        if i in seen:
            raise RuntimeError(f"stride produced duplicate index {i}; adjust selection")
        seen.add(i)
        out.append(rows[i])
    return out


def build_mini(*, archive_root: Path | None = None) -> Path:
    cfg = AppConfig()
    arch = Path(archive_root or cfg.archive_root)
    src_root = arch / "Drafts" / SOURCE_DRAFT
    mini_root = arch / "Drafts" / MINI_NAME
    if not src_root.is_dir():
        raise FileNotFoundError(f"missing source draft: {src_root}")

    cal_src = src_root / "calibrated" / "lights" / SETUP
    da_src = src_root / "detrended_aligned" / "lights" / SETUP
    ps_src = src_root / "platesolve" / SETUP
    for req, label in [
        (cal_src, "calibrated lights"),
        (da_src, "detrended_aligned lights"),
        (ps_src / "MASTERSTAR.fits", "MASTERSTAR.fits"),
        (ps_src / "variable_targets.csv", "variable_targets.csv"),
        (ps_src / "masterstars_full_match.csv", "masterstars_full_match.csv"),
        (ps_src / "per_frame_catalog_index.csv", "per_frame_catalog_index.csv"),
    ]:
        if not Path(req).exists():
            raise FileNotFoundError(f"fail-early: missing {label}: {req}")

    # Scope probe: local masters?
    raw_darks = src_root / "Raw" / "darks"
    raw_flats = src_root / "Raw" / "flats"
    n_darks = len(list(raw_darks.rglob("*.fits"))) if raw_darks.is_dir() else 0
    n_flats = len(list(raw_flats.rglob("*.fits"))) if raw_flats.is_dir() else 0
    n_raw_lights = len(list((src_root / "Raw" / "lights" / SETUP).glob("*.fits"))) if (
        src_root / "Raw" / "lights" / SETUP
    ).is_dir() else 0
    scope = (
        "photometry_ready"
        if (n_darks == 0 or n_flats == 0)
        else "from_raw"
    )

    selected = select_stride_frames(cal_src, da_src, N_FRAMES)
    basenames = [b for _d, b in selected]
    stems = [Path(b).stem for b in basenames]  # BO_CVn_Light_001

    if mini_root.exists():
        shutil.rmtree(mini_root)
    mini_root.mkdir(parents=True)

    inputs: list[dict] = []
    print(f"source: {src_root}")
    print(f"mini:   {mini_root}")
    print(f"scope:  {scope} (raw_lights={n_raw_lights}, local_darks={n_darks}, local_flats={n_flats})")
    print(f"setup:  {SETUP}  n={len(selected)}")
    print("selected frames (DATE-OBS, file):")
    for dobs, name in selected:
        print(f"  {dobs}  {name}")

    # draft metadata
    for meta_name in ("draft_manifest.json", "cal_diag.json"):
        meta_src = src_root / meta_name
        if meta_src.is_file():
            _copy_into(mini_root, meta_src, meta_name, inputs)

    # calibrated + optional raw lights for the selected set
    for name in basenames:
        _copy_into(mini_root, cal_src / name, f"calibrated/lights/{SETUP}/{name}", inputs)
        raw_src = src_root / "Raw" / "lights" / SETUP / name
        if raw_src.is_file():
            _copy_into(mini_root, raw_src, f"Raw/lights/{SETUP}/{name}", inputs)

    # detrended_aligned proc products
    for stem in stems:
        for ext in (".fits", ".csv"):
            src = da_src / f"proc_{stem}{ext}"
            if not src.is_file():
                raise FileNotFoundError(f"missing aligned product: {src}")
            _copy_into(
                mini_root,
                src,
                f"detrended_aligned/lights/{SETUP}/proc_{stem}{ext}",
                inputs,
            )

    # platesolve shared products (full catalogs; MASTERSTAR from parent night)
    for name in (
        "MASTERSTAR.fits",
        "variable_targets.csv",
        "masterstars_full_match.csv",
        "comparison_stars.csv",
        "alignment_report.csv",
        "photometry_plan.json",
        "variability_candidates.csv",
    ):
        src = ps_src / name
        if src.is_file():
            _copy_into(mini_root, src, f"platesolve/{SETUP}/{name}", inputs)

    # Filter per_frame_catalog_index to the selected stems only
    import pandas as pd

    idx_src = ps_src / "per_frame_catalog_index.csv"
    idx = pd.read_csv(idx_src, low_memory=False)
    # Match rows whose path/name contains one of the selected stems
    mask = False
    for stem in stems:
        col_match = False
        for col in idx.columns:
            if idx[col].dtype == object or str(idx[col].dtype).startswith("string"):
                col_match = col_match | idx[col].astype(str).str.contains(stem, regex=False)
        mask = mask | col_match
    if hasattr(mask, "sum") and int(mask.sum()) == 0:
        # Fallback: keep all rows if schema has no filename column match - still write full index
        print("WARN: could not filter per_frame_catalog_index by stem; copying full index")
        filtered = idx
    else:
        filtered = idx.loc[mask].copy()
        print(f"per_frame_catalog_index: {len(idx)} -> {len(filtered)} rows")
    idx_dst = mini_root / "platesolve" / SETUP / "per_frame_catalog_index.csv"
    idx_dst.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(idx_dst, index=False)
    inputs.append(
        {
            "rel": f"platesolve/{SETUP}/per_frame_catalog_index.csv",
            "sha256": _sha256_file(idx_dst),
            "bytes": idx_dst.stat().st_size,
            "note": "filtered_to_mini_stems",
        }
    )

    # Empty photometry output dir (regenerated by tests)
    (mini_root / "platesolve" / SETUP / "photometry").mkdir(parents=True, exist_ok=True)

    manifest = {
        "mini_name": MINI_NAME,
        "source_draft": SOURCE_DRAFT,
        "setup": SETUP,
        "n_frames": N_FRAMES,
        "selection": "even_stride_date_obs_first_included",
        "scope": scope,
        "scope_note": (
            "In-draft Raw/darks and Raw/flats have no local masters "
            "(CalibrationLibrary supplies them at import time). Mini is "
            "photometry-ready: calibrated + detrended_aligned proc products "
            "for 16 stride frames + parent platesolve catalogs/MASTERSTAR. "
            "Chain coverage matches session_baseline_check --full "
            "(run_full_photometry_pipeline)."
        ),
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frames": [{"date_obs": d, "file": n} for d, n in selected],
        "inputs": inputs,
    }
    # Manifest SHA over input file digests (order-stable)
    mh = hashlib.sha256()
    for item in sorted(inputs, key=lambda x: x["rel"]):
        mh.update(item["rel"].encode())
        mh.update(item["sha256"].encode())
    manifest["inputs_manifest_sha256"] = mh.hexdigest()

    man_path = mini_root / "p1_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"inputs_manifest_sha256: {manifest['inputs_manifest_sha256']}")
    print(f"wrote {man_path}")
    return mini_root


def main() -> int:
    build_mini()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
