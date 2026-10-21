#!/usr/bin/env python3
"""SAT-DIAG dry-run: read-only derivation/resolution per VYVAR_SAT_DIAG_SPEC.md.

Does not write draft data or mutate the pipeline. Reports what SAT-DIAG would decide.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from pipeline import (  # noqa: E402
    _effective_saturation_limit,
    _equipment_saturate_adu_from_db,
    _infer_sat_limit_from_bitpix,
    _saturate_limit_adu_from_header,
)

N_PILEUP_MIN = 100
PILEUP_RATIO = 10.0
SAT_DIAG_MAX_FRAMES = 30
HEADER_SAT_KEYS = ("SATURATE", "MAXLIN", "ESATUR", "LINLIMIT", "MAXADU", "DATAMAX")
ADMISSION_SAT_FRAC = 0.70
SATURATE_LIMIT_FRACTION = 0.85
KNOWN_GOOD_COMPS = ("1497771992240531712", "1499200223486564608")


@dataclass
class PileupResult:
    pileup_detected: bool
    v_ceiling: float | None
    n_at_ceiling: int
    n_at_ceiling_minus_1: int
    max_pixel: float
    frames_sampled: int
    bitpix_ceiling: float | None


@dataclass
class LimitResolution:
    sat_adu: float | None
    provenance: str
    header_value: float | None
    equipment_value: float | None
    derived_ceiling: float | None
    bitpix_ceiling: float | None
    compatibility_fired: bool
    refuted_source: str | None
    refuted_value: float | None
    current_pipeline_limit: float | None
    current_pipeline_source: str


def _image_adu_array(hdu: fits.PrimaryHDU) -> np.ndarray:
    """Image ADU as VYVAR uses for saturation (stored 0..65535 for unsigned 16-bit)."""
    hdr = hdu.header
    d = np.asarray(hdu.data, dtype=np.float64)
    try:
        bitpix = int(hdr.get("BITPIX", 0))
        bzero = float(hdr.get("BZERO", 0.0))
        bscale = float(hdr.get("BSCALE", 1.0))
    except (TypeError, ValueError):
        return d
    if bitpix == 16 and abs(bzero - 32768.0) < 1.0 and abs(bscale - 1.0) < 1e-9:
        # QHY / unsigned 16-bit: pipeline and histogram audits use stored ADU 0..65535.
        return d
    bzero = float(hdr.get("BZERO", 0.0))
    bscale = float(hdr.get("BSCALE", 1.0))
    return d * bscale + bzero


def _bitpix_container(hdr: fits.Header) -> float | None:
    return _infer_sat_limit_from_bitpix(hdr)


def _sample_raw_lights(draft_dir: Path, max_frames: int = SAT_DIAG_MAX_FRAMES) -> list[Path]:
    for sub in ("Raw/lights", "calibrated/lights"):
        root = draft_dir / sub
        if not root.is_dir():
            continue
        files = sorted(root.rglob("*.fits"))
        if not files:
            continue
        if len(files) <= max_frames:
            return files
        idx = np.linspace(0, len(files) - 1, max_frames, dtype=int)
        return [files[i] for i in idx]
    return []


def derive_ceiling_from_raw(paths: list[Path]) -> PileupResult:
    if not paths:
        return PileupResult(False, None, 0, 0, float("nan"), 0, None)

    counts: Counter[int] = Counter()
    max_px = -math.inf
    bitpix_ceil: float | None = None
    for fp in paths:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
            if bitpix_ceil is None:
                bitpix_ceil = _bitpix_container(hdr)
            arr = _image_adu_array(hdul[0])
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            max_px = max(max_px, float(np.max(finite)))
            vals, cnts = np.unique(finite.astype(np.int64), return_counts=True)
            for v, c in zip(vals, cnts, strict=False):
                counts[int(v)] += int(c)

    if not counts:
        return PileupResult(False, None, 0, 0, max_px, len(paths), bitpix_ceil)

    v_max = max(counts)
    n_max = counts[v_max]
    # Shoulder: highest occupied bin strictly below v_max (65534 may be empty at ceiling).
    lower_vals = [v for v in counts if v < v_max]
    n_prev = counts[max(lower_vals)] if lower_vals else 0
    pileup = False
    if n_max >= N_PILEUP_MIN and bitpix_ceil is not None and v_max >= int(bitpix_ceil) - 1:
        if n_prev <= 0:
            pileup = True
        elif n_max >= PILEUP_RATIO * n_prev:
            pileup = True
    if pileup:
        return PileupResult(True, float(v_max), n_max, n_prev, max_px, len(paths), bitpix_ceil)
    return PileupResult(False, None, n_max, n_prev, max_px, len(paths), bitpix_ceil)


def _compatible(stated: float | None, max_pixel: float) -> bool:
    if stated is None or not math.isfinite(stated) or stated <= 0:
        return False
    return float(stated) >= float(max_pixel)


def resolve_sat_limit(
    *,
    hdr: fits.Header,
    pileup: PileupResult,
    equipment_adu: float | None,
) -> LimitResolution:
    header_val = _saturate_limit_adu_from_header(hdr)
    for dk in ("DATAMAX", "MAXPIX"):
        if header_val is None and dk in hdr:
            try:
                v = float(hdr[dk])
                if math.isfinite(v) and v > 0:
                    header_val = v
                    break
            except (TypeError, ValueError):
                pass

    derived = pileup.v_ceiling if pileup.pileup_detected else None
    bitpix = pileup.bitpix_ceiling or _bitpix_container(hdr)
    max_px = pileup.max_pixel

    refuted_source = None
    refuted_value = None
    compatibility_fired = False

    def try_stated(value: float | None, source: str) -> tuple[float | None, str | None]:
        nonlocal compatibility_fired, refuted_source, refuted_value
        if value is None:
            return None, None
        if _compatible(value, max_px):
            return float(value), source
        compatibility_fired = True
        refuted_source = source
        refuted_value = float(value)
        return None, None

    win_val: float | None = None
    prov = "none"

    v, src = try_stated(header_val, "HEADER")
    if v is not None:
        win_val, prov = v, src

    if win_val is None:
        v, src = try_stated(equipment_adu, "EQUIPMENT")
        if v is not None:
            win_val, prov = v, src

    if win_val is None and derived is not None:
        win_val, prov = float(derived), "DERIVED"

    if win_val is None and bitpix is not None:
        win_val, prov = float(bitpix), "DERIVED_NO_PILEUP" if not pileup.pileup_detected else "BITPIX"

        if compatibility_fired and derived is not None:
            win_val = float(derived)
            prov = "CONFLICT_DERIVED"

    cur_lim, cur_src = _effective_saturation_limit(
        hdr, fallback_adu=None, equipment_saturate_adu=equipment_adu
    )

    return LimitResolution(
        sat_adu=win_val,
        provenance=prov,
        header_value=header_val,
        equipment_value=equipment_adu,
        derived_ceiling=derived,
        bitpix_ceiling=bitpix,
        compatibility_fired=compatibility_fired,
        refuted_source=refuted_source,
        refuted_value=refuted_value,
        current_pipeline_limit=cur_lim,
        current_pipeline_source=cur_src,
    )


def _read_manifest(draft_dir: Path) -> dict[str, Any]:
    p = draft_dir / "draft_manifest.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _obs_group_dir(draft_dir: Path) -> Path | None:
    ps = draft_dir / "platesolve"
    if not ps.is_dir():
        return None
    subs = [d for d in ps.iterdir() if d.is_dir() and not d.name.startswith("_")]
    return subs[0] if len(subs) == 1 else (subs[0] if subs else None)


def _admission_threshold(sat_adu: float | None) -> float | None:
    if sat_adu is None:
        return None
    return float(sat_adu) * SATURATE_LIMIT_FRACTION * (ADMISSION_SAT_FRAC / SATURATE_LIMIT_FRACTION)


def _comp_pool_report(draft_dir: Path, sat_adu: float | None) -> dict[str, Any]:
    og = _obs_group_dir(draft_dir)
    if og is None:
        return {"error": "no obs_group"}
    ms_path = og / "masterstars_full_match.csv"
    if not ms_path.is_file():
        return {"error": f"missing {ms_path.name}"}
    import pandas as pd

    df = pd.read_csv(ms_path)
    if "peak_max_adu" not in df.columns:
        return {"error": "no peak_max_adu column"}

    adm = _admission_threshold(sat_adu)
    peak = pd.to_numeric(df["peak_max_adu"], errors="coerce")
    id_col = "catalog_id" if "catalog_id" in df.columns else "name"
    ids = df[id_col].astype(str)

    zone_sat = df.get("zone", pd.Series("", index=df.index)).astype(str).str.lower().eq("saturated")
    likely = df.get("likely_saturated", pd.Series(False, index=df.index))
    if likely.dtype != bool:
        likely = likely.astype(str).str.lower().isin(("true", "1", "yes"))

    n_total = len(df)
    if adm is not None:
        over_adm = peak > adm
    else:
        over_adm = pd.Series(False, index=df.index)

    pool_ok = ~(zone_sat | likely | over_adm.fillna(False))
    n_pool = int(pool_ok.sum())

    comp_rows = {}
    for cid in KNOWN_GOOD_COMPS:
        m = ids == str(cid)
        if not m.any():
            m = ids.str.replace("Gaia_DR3_", "", regex=False).str.strip() == str(cid)
        if not m.any():
            comp_rows[cid] = {"found": False}
            continue
        row = df.loc[m].iloc[0]
        pk = float(row.get("peak_max_adu", float("nan")))
        comp_rows[cid] = {
            "found": True,
            "peak_max_adu": pk,
            "zone": str(row.get("zone", "")),
            "pass_admission_new_limit": bool(math.isfinite(pk) and adm is not None and pk <= adm),
            "pass_admission_old_16384": bool(math.isfinite(pk) and pk <= 16384 * 0.85 * (ADMISSION_SAT_FRAC / SATURATE_LIMIT_FRACTION)),
        }

    return {
        "masterstars_rows": n_total,
        "comparison_pool_simulated": n_pool,
        "admission_threshold_adu": adm,
        "known_comps": comp_rows,
        "zone_saturated_count": int(zone_sat.sum()),
        "likely_saturated_count": int(likely.sum()),
        "note": "Uses static masterstars CSV peaks; re-run photometry needed for live flags.",
    }


def analyze_draft(draft_id: int, equipment_adu: float | None) -> dict[str, Any]:
    draft_dir = REPO / "Archive" / "Drafts" / f"draft_{draft_id:06d}"
    manifest = _read_manifest(draft_dir)
    rig = manifest.get("rig") or {}
    equip_id = rig.get("equipment_id")

    raw_paths = _sample_raw_lights(draft_dir)
    pileup = derive_ceiling_from_raw(raw_paths)

    hdr: fits.Header = fits.Header()
    xbin = ybin = None
    if raw_paths:
        with fits.open(raw_paths[0], memmap=False) as hdul:
            hdr = hdul[0].header
            xbin = hdr.get("XBINNING")
            ybin = hdr.get("YBINNING")

    eq = equipment_adu
    if eq is None and equip_id is not None:
        eq = _equipment_saturate_adu_from_db(int(equip_id))

    lim = resolve_sat_limit(hdr=hdr, pileup=pileup, equipment_adu=eq)

    return {
        "draft_id": draft_id,
        "draft_dir": str(draft_dir.relative_to(REPO)),
        "equipment_id": equip_id,
        "xbinning": xbin,
        "ybinning": ybin,
        "raw_frames_sampled": len(raw_paths),
        "pileup": asdict(pileup),
        "limit": asdict(lim),
        "comp_pool": _comp_pool_report(draft_dir, lim.sat_adu),
        "comp_pool_at_current_pipeline": _comp_pool_report(draft_dir, lim.current_pipeline_limit),
    }


def run_failure_modes() -> dict[str, Any]:
    """Synthetic cases for spec survival claims."""
    hdr_raw = fits.Header()
    hdr_raw["BITPIX"] = 16
    hdr_raw["BZERO"] = 32768
    hdr_raw["BSCALE"] = 1.0

    # Simulated pile-up field: many pixels at 65535
    rng = np.random.default_rng(0)
    data_pileup = rng.integers(1000, 30000, size=(512, 512)).astype(np.float64)
    data_pileup[100:120, 100:120] = 65535

    paths_tmp = REPO / "tmp" / "_sat_diag_synth"
    paths_tmp.mkdir(parents=True, exist_ok=True)
    synth_path = paths_tmp / "synth_pileup.fits"
    fits.writeto(synth_path, data_pileup.astype(np.int16), hdr_raw, overwrite=True)

    pileup = derive_ceiling_from_raw([synth_path])

    out: dict[str, Any] = {}

    # equipment absent
    out["equipment_absent"] = asdict(
        resolve_sat_limit(hdr=hdr_raw, pileup=pileup, equipment_adu=None)
    )

    # no header keyword
    out["no_header_keyword"] = asdict(
        resolve_sat_limit(hdr=hdr_raw, pileup=pileup, equipment_adu=None)
    )

    # header refuted (16384 but data has 65535)
    hdr_bad = hdr_raw.copy()
    hdr_bad["SATURATE"] = 16384
    out["header_refuted"] = asdict(
        resolve_sat_limit(hdr=hdr_bad, pileup=pileup, equipment_adu=None)
    )

    # no pile-up: shallow field
    data_shallow = rng.integers(1000, 12000, size=(512, 512)).astype(np.float64)
    shallow_path = paths_tmp / "synth_shallow.fits"
    fits.writeto(shallow_path, data_shallow.astype(np.int16), hdr_raw, overwrite=True)
    pileup_shallow = derive_ceiling_from_raw([shallow_path])
    out["no_pileup"] = {
        "pileup": asdict(pileup_shallow),
        "limit": asdict(
            resolve_sat_limit(hdr=hdr_raw, pileup=pileup_shallow, equipment_adu=None)
        ),
    }

    # equipment row wrong binning assumption (16384 stated, data at 65535)
    out["equipment_refuted"] = asdict(
        resolve_sat_limit(hdr=hdr_raw, pileup=pileup, equipment_adu=16384.0)
    )

    # calibrated float mistake: BITPIX -32, values above 65535
    hdr_float = fits.Header()
    hdr_float["BITPIX"] = -32
    data_float = np.full((256, 256), 69000.0, dtype=np.float32)
    float_path = paths_tmp / "synth_aligned_float.fits"
    fits.writeto(float_path, data_float, hdr_float, overwrite=True)
    with fits.open(float_path, memmap=False) as hdul:
        arr = _image_adu_array(hdul[0])
        max_f = float(np.max(arr))
    pileup_float = derive_ceiling_from_raw([float_path])
    _plim, _psrc = _effective_saturation_limit(
        hdr_float, fallback_adu=None, equipment_saturate_adu=16384.0
    )
    out["calibrated_float_mistake"] = {
        "max_pixel": max_f,
        "pileup": asdict(pileup_float),
        "pipeline_current": {"limit": _plim, "source": _psrc},
        "sat_diag": asdict(
            resolve_sat_limit(hdr=hdr_float, pileup=pileup_float, equipment_adu=16384.0)
        ),
        "warning": "float frame has no pile-up pattern; BITPIX=-32 gives no container bound",
    }

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="SAT-DIAG dry-run (read-only)")
    ap.add_argument("--drafts", type=int, nargs="*", default=[435, 509, 510])
    ap.add_argument("--json-out", type=Path, default=REPO / "tmp" / "_sat_diag_dry_run.json")
    ap.add_argument("--failure-modes", action="store_true", default=True)
    args = ap.parse_args()

    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    equip_sat = db.get_equipment_saturation_adu(1)

    report: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO
        ).strip(),
        "equipment_1_saturate_adu_db": equip_sat,
        "drafts": [analyze_draft(d, equip_sat) for d in args.drafts],
        "drafts_skipped": [],
    }

    for d in (436, 437):
        dd = REPO / "Archive" / "Drafts" / f"draft_{d:06d}"
        if not _sample_raw_lights(dd):
            report["drafts_skipped"].append({"draft_id": d, "reason": "no raw/calibrated lights on disk"})

    if args.failure_modes:
        report["failure_modes"] = run_failure_modes()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
