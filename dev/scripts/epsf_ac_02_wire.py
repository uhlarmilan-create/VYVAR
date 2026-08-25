#!/usr/bin/env python3
"""EPSF-AC-02-WIRE: P4 stamp + LC-PIN regen + BO/FW meters on draft 516.

Does not re-fit. Inverts stored F6 AC on proc sidecars (ADDITIVE-01 per file),
rewrites internal PSF LCs, measures pin-rule meters. Production ePSF FITS+meta,
aperture LCs, AAVSO, and VarAstro are hash-guarded.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from epsf_psf_merge import stamp_p4_none_science_sidecars  # noqa: E402
from psf_internal_lc import write_internal_psf_lightcurves  # noqa: E402

DRAFT = REPO / "Archive" / "Drafts" / "draft_000516"
PS = DRAFT / "platesolve" / "NoFilter_60_2"
FRAMES = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"
PHOT = PS / "photometry"
LC_DIR = PHOT / "lightcurves"
OUT = REPO / "dev" / "results" / "session_20260824_epsf_ac_02_wire"
PROD_EPSF = PS / "masterstar_epsf.fits"
PROD_META = PS / "masterstar_epsf_meta.json"
BO_CVN = "1498613634033133184"
FW_CVN = "1497343732462852864"
PROD_EPSF_SHA = "172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20"
A3_BO_RMS_MMAG = 614.2695148951224
A1_SLOPE_TERM_MMAG = 76.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO)).replace("\\", "/")


def must_not_change_files() -> list[Path]:
    files = [PROD_EPSF, PROD_META]
    files.extend(sorted(p for p in LC_DIR.glob("lightcurve_*.csv") if "_psf" not in p.name and "_adaptive" not in p.name))
    aavso = PHOT / "lightcurves_reports" / "aavso"
    varastro = PHOT / "lightcurves_reports" / "varastro"
    if aavso.is_dir():
        files.extend(sorted(aavso.glob("*.txt")))
    if varastro.is_dir():
        files.extend(sorted(varastro.glob("*.txt")))
    return [p for p in files if p.is_file()]


def expected_changed_files() -> list[Path]:
    files = sorted(LC_DIR.glob("lightcurve_*_psf.csv"))
    procs = sorted(FRAMES.glob("proc_*.csv"))
    if procs:
        files.extend(procs[:5])
        files.append(procs[-1])
    return [p for p in files if p.is_file()]


def snapshot_hashes(paths: list[Path], label: str) -> dict[str, str]:
    out = {_rel(p): _sha(p) for p in paths}
    (OUT / f"{label}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="ascii")
    return out


def parse_header_kv(path: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        body = line[1:].strip()
        if "=" not in body:
            continue
        k, v = body.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv


def meters_for_target(cid: str) -> dict[str, Any]:
    path = LC_DIR / f"lightcurve_{cid}_psf.csv"
    if not path.is_file():
        return {"catalog_id": cid, "error": "missing_psf_lc"}
    hdr = parse_header_kv(path)
    df = pd.read_csv(path, comment="#")
    psf = pd.to_numeric(df.get("psf_delta_mag"), errors="coerce").to_numpy(dtype=np.float64)
    ap = pd.to_numeric(df.get("delta_mag"), errors="coerce").to_numpy(dtype=np.float64)
    reason = df["psf_epoch_drop_reason"].astype(str) if "psf_epoch_drop_reason" in df.columns else pd.Series([""] * len(df))
    pin_drop = np.array(
        [str(r).strip().lower() not in ("", "nan", "none") for r in reason.tolist()],
        dtype=bool,
    )
    n = int(len(df))
    n_drop = int(pin_drop.sum())
    n_full = int(n - n_drop)
    both = (~pin_drop) & np.isfinite(psf) & np.isfinite(ap)
    residual = psf[both] - ap[both]
    n_both = int(both.sum())
    med = float(np.median(residual)) if n_both else float("nan")
    rms = float(np.sqrt(np.mean(residual**2))) if n_both else float("nan")
    rms_vs_abs_med = bool(
        n_both > 0
        and abs(rms - abs(med)) <= 0.25 * max(abs(med), abs(rms), 1e-12)
    )
    return {
        "catalog_id": cid,
        "n_epochs": n,
        "n_full_membership": n_full,
        "n_dropped_pin": n_drop,
        "coverage_pin_survive": (n_full / n) if n else float("nan"),
        "n_finite_pairs": n_both,
        "level_offset_mag": med,
        "level_offset_mmag": med * 1000.0 if math.isfinite(med) else float("nan"),
        "rms_mag": rms,
        "rms_mmag": rms * 1000.0 if math.isfinite(rms) else float("nan"),
        "rms_vs_abs_median": rms_vs_abs_med,
        "header_n_full": hdr.get("psf_lc_n_epochs_full"),
        "header_n_dropped_pin": hdr.get("psf_lc_n_epochs_dropped_pin"),
        "header_level_offset": hdr.get("psf_ap_level_offset_mag"),
        "header_policy": hdr.get("psf_ac_policy"),
        "psf_lc_sha256": _sha(path),
    }


def residual_table(cid: str, out_csv: Path) -> None:
    path = LC_DIR / f"lightcurve_{cid}_psf.csv"
    df = pd.read_csv(path, comment="#")
    psf = pd.to_numeric(df.get("psf_delta_mag"), errors="coerce")
    ap = pd.to_numeric(df.get("delta_mag"), errors="coerce")
    reason = df["psf_epoch_drop_reason"].astype(str) if "psf_epoch_drop_reason" in df.columns else ""
    pin = [str(r).strip().lower() not in ("", "nan", "none") for r in pd.Series(reason).tolist()]
    out = pd.DataFrame(
        {
            "source_file": df.get("source_file"),
            "psf_delta_mag": psf,
            "delta_mag": ap,
            "residual": psf - ap,
            "psf_epoch_drop_reason": reason,
            "full_membership": [not p for p in pin],
        }
    )
    out.to_csv(out_csv, index=False)


def copy_context() -> dict[str, Any]:
    ctx = REPO / "dev" / "results" / "context" / "session_20260824_epsf_ac_02_wire"
    ctx.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    blobs: dict[str, str] = {}
    for p in sorted(OUT.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".csv", ".json", ".txt", ".md"}:
            shutil.copy2(p, ctx / p.name)
            copied.append(p.name)
        else:
            blobs[p.name] = _sha(p)
    manifest = {"copied_text": copied, "blobs_not_copied": blobs}
    (ctx / "BLOB_SHA_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    return manifest


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    epsf_sha = _sha(PROD_EPSF)
    (OUT / "g4_epsf_sha.txt").write_text(epsf_sha + "\n", encoding="ascii")
    if epsf_sha != PROD_EPSF_SHA:
        raise SystemExit(f"G4 FAIL: production ePSF SHA {epsf_sha}")

    h_guard_b = snapshot_hashes(must_not_change_files(), "hashes_must_not_change_before")
    h_chg_b = snapshot_hashes(expected_changed_files(), "hashes_expected_changed_before")

    stamp = stamp_p4_none_science_sidecars(FRAMES, platesolve_dir=PS)
    (OUT / "stamp_summary.json").write_text(json.dumps(stamp, indent=2, default=str) + "\n", encoding="ascii")

    lc = write_internal_psf_lightcurves(platesolve_dir=PS, frames_root=FRAMES)
    per_target: list[dict[str, Any]] = []
    for written in lc.get("written") or []:
        p = Path(written)
        hdr = parse_header_kv(p)
        cid = p.stem.replace("lightcurve_", "", 1).replace("_psf", "")
        per_target.append(
            {
                "catalog_id": cid,
                "n_epochs_full": hdr.get("psf_lc_n_epochs_full"),
                "n_epochs_dropped_pin": hdr.get("psf_lc_n_epochs_dropped_pin"),
                "level_offset_mag": hdr.get("psf_ap_level_offset_mag"),
                "policy": hdr.get("psf_ac_policy"),
            }
        )
    lc_summary = {
        "n_written": lc.get("n_written"),
        "n_skipped": lc.get("n_skipped"),
        "per_target": per_target,
    }
    (OUT / "lc_regen.json").write_text(json.dumps(lc_summary, indent=2) + "\n", encoding="ascii")
    pd.DataFrame(per_target).to_csv(OUT / "per_target_pin_counts.csv", index=False)

    bo = meters_for_target(BO_CVN)
    fw = meters_for_target(FW_CVN)
    residual_table(BO_CVN, OUT / "bo_cvn_residuals.csv")
    residual_table(FW_CVN, OUT / "fw_cvn_residuals.csv")
    (OUT / "bo_meters.json").write_text(json.dumps(bo, indent=2) + "\n", encoding="ascii")
    (OUT / "fw_meters.json").write_text(json.dumps(fw, indent=2) + "\n", encoding="ascii")

    h_guard_a = snapshot_hashes(must_not_change_files(), "hashes_must_not_change_after")
    h_chg_a = snapshot_hashes(expected_changed_files(), "hashes_expected_changed_after")
    guard_ok = h_guard_b == h_guard_a
    changed = {k: (h_chg_b.get(k), h_chg_a.get(k)) for k in sorted(set(h_chg_b) | set(h_chg_a)) if h_chg_b.get(k) != h_chg_a.get(k)}
    summary = {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "g4_epsf_sha": epsf_sha,
        "g4_ok": epsf_sha == PROD_EPSF_SHA,
        "hash_guard_ok": guard_ok,
        "n_must_not_change": len(h_guard_b),
        "n_expected_changed_differ": len(changed),
        "stamp_n": stamp.get("written"),
        "lc_n_written": lc.get("n_written"),
        "bo": bo,
        "fw": fw,
        "a3_baseline_bo_rms_mmag": A3_BO_RMS_MMAG,
        "a1_slope_term_mmag": A1_SLOPE_TERM_MMAG,
        "merge_meta": stamp.get("merge_meta"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    (OUT / "expected_changed_diff_keys.json").write_text(
        json.dumps(sorted(changed.keys()), indent=2) + "\n", encoding="ascii"
    )
    copy_context()
    if not guard_ok:
        drift = {k: (h_guard_b.get(k), h_guard_a.get(k)) for k in h_guard_b if h_guard_b[k] != h_guard_a.get(k)}
        raise SystemExit(f"hash guard FAIL: {list(drift)[:8]}")
    print(json.dumps({"ok": True, **{k: summary[k] for k in ("elapsed_s", "lc_n_written", "stamp_n", "hash_guard_ok")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
