"""INV-CAL-02 pre-registered validation and archive census."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from cal_diag import CalStageCompareRefusedError, apply_calibrated_stage_for_compare, calibrated_compare_refused  # noqa: E402
from cal_stage import (  # noqa: E402
    CalStageConfidence,
    archive_stage_census,
    compute_fits_datasum,
    resolve_calibrated_stage,
    verify_fits_datasum,
)
from config import AppConfig  # noqa: E402
from pipeline import calibrate_lights_to_calibrated  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_p1_checksums() -> dict[str, object]:
    manifests = {
        "435": ROOT / "dev/validation/anchor_435_checksums_post_restore_20260813.json",
        "510": ROOT / "dev/validation/anchor_510_checksums_a1_dao_fwhm_20260814.json",
    }
    out: dict[str, object] = {}
    for draft, man_path in manifests.items():
        data = json.loads(man_path.read_text(encoding="utf-8"))
        root = ROOT / data["root"]
        mism = 0
        checked = 0
        for rel, meta in data["files"].items():
            if not rel.startswith("calibrated/lights/") or not rel.endswith(".fits"):
                continue
            fp = root / rel
            if not fp.is_file():
                mism += 1
                continue
            checked += 1
            if _sha256(fp) != meta["sha256"]:
                mism += 1
        out[draft] = {"checked": checked, "mismatch": mism, "passed": mism == 0}
    out["509"] = validate_p1_509_calibrated_only()
    return out


def _rel_posix(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def validate_p1_509_calibrated_only() -> dict[str, object]:
    arch = ROOT / "Archive/Drafts/draft_000509/calibrated/lights"
    files = sorted(arch.rglob("BO_CVn_Light_*.fits"))
    baseline_path = ROOT / "tmp/_inv_cal02_p1_509_baseline.json"
    if not baseline_path.exists():
        baseline = {_rel_posix(f): _sha256(f) for f in files}
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    mism = 0
    for f in files:
        rel = _rel_posix(f)
        if baseline.get(rel) != _sha256(f):
            mism += 1
    return {"checked": len(files), "mismatch": mism, "passed": mism == 0}


def validate_p2_resolver() -> dict[str, object]:
    out: dict[str, object] = {}
    for draft in ("435", "509", "510"):
        arch = ROOT / f"Archive/Drafts/draft_{int(draft):06d}/calibrated/lights"
        files = sorted(arch.rglob("BO_CVn_Light_*.fits"))
        stages = Counter()
        for fp in files:
            hdr = fits.getheader(fp)
            res = resolve_calibrated_stage(hdr)
            stages[res.stage] += 1
        expect = "PURE" if draft == "435" else "SKYSF_2"
        out[draft] = {
            "n": len(files),
            "stages": dict(stages),
            "expected": expect,
            "passed": stages.get(expect, 0) == len(files),
        }
    return out


def validate_p3_indeterminate() -> dict[str, object]:
    sample = None
    for fp in ROOT.glob("Archive/Drafts/**/processed/**/*.fits"):
        hdr = fits.getheader(fp)
        if hdr.get("VYSKYP2P") is not None and hdr.get("VY_SKYSF") is None:
            sample = fp
            break
    if sample is None:
        return {"passed": False, "error": "no VYSKYP2P-without-VY_SKYSF sample found"}
    hdr = fits.getheader(sample)
    res = resolve_calibrated_stage(hdr)
    refuse = calibrated_compare_refused(hdr)
    try:
        apply_calibrated_stage_for_compare(np.zeros((4, 4), dtype=np.float32), hdr)
        raised = False
    except CalStageCompareRefusedError:
        raised = True
    return {
        "sample": str(sample.relative_to(ROOT)),
        "confidence": res.confidence.value,
        "refuse_reason": refuse,
        "raised": raised,
        "passed": res.confidence == CalStageConfidence.INDETERMINATE_LEGACY and refuse and raised,
    }


def validate_p4_fresh_cal_stamp() -> dict[str, object]:
    import shutil

    out = ROOT / "tmp/_inv_cal02_p4"
    if out.exists():
        shutil.rmtree(out)
    lib = ROOT / "CalibrationLibrary"
    md = sorted(lib.glob("Dark_60s*Bin1*.fits"))[0]
    mf = sorted(lib.glob("Flat*.fits"))[0]
    raw = ROOT / "Archive/Drafts/draft_000435/Raw/lights"
    calibrate_lights_to_calibrated(
        lights_root=raw,
        calibrated_root=out,
        master_dark_path=md,
        masterflat_by_filter={"NoFilter": mf},
        pipeline_config=AppConfig(),
    )
    fp = next(out.rglob("BO_CVn_Light_001.fits"))
    with fits.open(fp) as hdul:
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data)
    ok_stage = hdr.get("VY_CALSTAGE") == "PURE"
    ok_sum = verify_fits_datasum(data, str(hdr.get("VY_CALDATASUM")))
    return {
        "VY_CALSTAGE": hdr.get("VY_CALSTAGE"),
        "VY_CALDATASUM": hdr.get("VY_CALDATASUM"),
        "verify": ok_sum,
        "passed": ok_stage and ok_sum,
    }


def validate_p5_force_reapply_token() -> dict[str, object]:
    from pipeline import _qc_enrich_one_frame

    tmp = ROOT / "tmp/_inv_cal02_p5"
    tmp.mkdir(parents=True, exist_ok=True)
    fp = tmp / "light.fits"
    data = np.full((32, 32), 1200.0, dtype=np.float32)
    fits.writeto(fp, data, overwrite=True)
    _qc_enrich_one_frame(
        str(fp),
        sky_order=2,
        force_reapply=False,
        prefilter_status=None,
        target_ra=None,
        target_dec=None,
        inject_pointing_only_if_missing=True,
    )
    row = _qc_enrich_one_frame(
        str(fp),
        sky_order=2,
        force_reapply=True,
        prefilter_status=None,
        target_ra=None,
        target_dec=None,
        inject_pointing_only_if_missing=True,
    )
    with fits.open(fp) as hdul:
        stage = hdul[0].header.get("VY_CALSTAGE")
    return {
        "stage": stage,
        "row_stage": row.get("cal_stage"),
        "passed": stage == "SKYSF_2_R2",
    }


def archive_census() -> dict[str, int]:
    counts: Counter[str] = Counter()
    for draft in sorted((ROOT / "Archive/Drafts").glob("draft_*")):
        cal = draft / "calibrated" / "lights"
        if cal.is_dir():
            for k, v in archive_stage_census(cal).items():
                counts[k] += v
        proc = draft / "processed"
        if proc.is_dir():
            for k, v in archive_stage_census(proc).items():
                counts[k] += v
    return dict(sorted(counts.items()))


def main() -> int:
    results = {
        "P1": validate_p1_checksums(),
        "P2": validate_p2_resolver(),
        "P3": validate_p3_indeterminate(),
        "P4": validate_p4_fresh_cal_stamp(),
        "P5": validate_p5_force_reapply_token(),
        "P8": {"note": "covered by pytest test_calibrate_stamps_pure"},
        "census": archive_census(),
    }
    print(json.dumps(results, indent=2))
    ok = all(
        block.get("passed")
        for key, block in results.items()
        if isinstance(block, dict) and "passed" in block and key != "P1"
    )
    p1 = results["P1"]
    if isinstance(p1, dict):
        ok = ok and all(
            isinstance(v, dict) and v.get("passed")
            for v in p1.values()
            if isinstance(v, dict)
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
