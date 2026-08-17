"""Headless Phase 0+1+2A for draft 515 on existing calibrated/aligned products.

PFS-SEMANTICS-01 / SAT-RERANK-01: full 97-target rebuild with INV-SAT-LIMIT
armed on the catalog and per-frame saturation ON via AppConfig instance
override (not a persisted config.json key; registry stays 291).
RUN-HARDEN-01 harness (progress every target). Does not re-calibrate or re-platesolve.

UTF-8 file log is opened by this process so [COMP] lines survive (do not wrap
this script with PowerShell *>).
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from invariants_runtime import (  # noqa: E402
    STAGE_ORDER,
    load_pipeline_meta,
    save_pipeline_meta,
)
from photometry_core import run_full_photometry_pipeline  # noqa: E402

SAT_JSON = ROOT / "dev" / "results" / "SAT_LIMIT_01_summary.json"
LOG_PATH = ROOT / "tmp" / "draft_515_pfs_semantics_01.log"


class _Tee:
    """Write to console and a UTF-8 file. Avoids PowerShell UTF-16 wrapping."""

    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            try:
                st.write(s)
            except Exception:  # noqa: BLE001
                continue
        self.flush()
        return len(s) if isinstance(s, str) else 0

    def flush(self) -> None:
        for st in self.streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                continue

    def isatty(self) -> bool:
        return False


def _setup_utf8_file_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logf = path.open("w", encoding="utf-8", newline="\n")
    sys.stdout = _Tee(sys.__stdout__, logf)
    sys.stderr = _Tee(sys.__stderr__, logf)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.StreamHandler(logf)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return logf


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _saturated_ids() -> list[str]:
    data = json.loads(SAT_JSON.read_text(encoding="utf-8"))
    ids = data.get("b4", {}).get("reclassify", {}).get("saturated_catalog_ids") or []
    return [str(x).strip() for x in ids]


def _preflight_catalog(ms_csv: Path, sat_ids: list[str]) -> None:
    import pandas as pd

    ms = pd.read_csv(ms_csv, dtype={"catalog_id": str}, low_memory=False)
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    sat = ms["is_saturated"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
    flagged = set(ms.loc[sat, "catalog_id"])
    missing = [i for i in sat_ids if i not in flagged]
    extra_n = int(sat.sum())
    print(f"PREFLIGHT n_is_saturated={extra_n} expected={len(sat_ids)}", flush=True)
    if missing:
        raise SystemExit(
            f"PREFLIGHT FAIL: {len(missing)} SAT-LIMIT IDs not flagged: {missing[:8]}"
        )
    print("PREFLIGHT PASS: all SAT-LIMIT-01 saturated IDs flagged in catalog", flush=True)


def _trim_dag_before_phase01_restamp(phot: Path) -> None:
    """Drop leftover phase2a/postprocess stamps so INV-DAG-01 can re-stamp.

    Known friction: a prior run leaves seq=7 postprocess in pipeline_meta.json;
    this rebuild's phase01/phase2a stamps then fail closed (seq goes backwards).
    Same trim as wide_err_03b_reexport, cut before phase01 so both stamps land.
    """
    meta = load_pipeline_meta(phot)
    stages = meta.get("stages") if isinstance(meta.get("stages"), list) else []
    cut = STAGE_ORDER.index("phase01")
    kept = [
        s
        for s in stages
        if isinstance(s, dict)
        and str(s.get("name") or "") in STAGE_ORDER
        and STAGE_ORDER.index(str(s.get("name"))) < cut
    ]
    meta["stages"] = kept
    save_pipeline_meta(phot, meta)
    print(
        f"DAG_TRIM remaining={len(kept)} names={[s.get('name') for s in kept]}",
        flush=True,
    )


def main() -> int:
    logf = _setup_utf8_file_log(LOG_PATH)
    draft_id = 515
    setup = "NoFilter_60_2"
    draft = ROOT / "Archive" / "Drafts" / f"draft_{draft_id:06d}"
    og = draft / "platesolve" / setup
    phot = og / "photometry"
    pf_dir = draft / "detrended_aligned" / "lights" / setup
    sha = _git_sha()
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("HARNESS draft_515_headless_phase012a PFS-SEMANTICS-01", flush=True)
    print(f"GIT_SHA {sha}", flush=True)
    print(f"START_UTC {started}", flush=True)
    print(f"DRAFT {draft}", flush=True)
    print(f"LOG_UTF8 {LOG_PATH}", flush=True)

    sat_ids = _saturated_ids()
    _preflight_catalog(og / "masterstars_full_match.csv", sat_ids)
    _trim_dag_before_phase01_restamp(phot)
    lc_dir = phot / "lightcurves"
    if lc_dir.is_dir():
        n_lc = 0
        for p in lc_dir.glob("lightcurve_*.csv"):
            p.unlink()
            n_lc += 1
        print(f"LC_TRIM n={n_lc}", flush=True)

    cfg = AppConfig()
    print(
        f"PFS_AFTER_LOAD {bool(cfg.per_frame_saturation_enabled)} "
        "(persisted config.json; AppConfig default is false)",
        flush=True,
    )
    # Existing per-run override: mutate the loaded AppConfig instance.
    # Not a new registry key (291 stays 291). Snapshot in pipeline_meta captures it.
    cfg.per_frame_saturation_enabled = True
    print(f"PFS_RUN_OVERRIDE {bool(cfg.per_frame_saturation_enabled)}", flush=True)
    db = VyvarDatabase(Path(cfg.database_path))
    marks: dict[str, float] = {}

    def _prog(msg: str) -> None:
        elapsed = time.perf_counter() - t0
        print(f"[{elapsed:8.1f}s] {msg}", flush=True)
        s = str(msg)
        if s.startswith("Faza 0:") and "phase0_start" not in marks:
            marks["phase0_start"] = elapsed
        if s.startswith("Faza 0 hotova") and "phase0_end" not in marks:
            marks["phase0_end"] = elapsed
        if ("Faza 1:" in s or s.startswith("Phase 1:")) and "phase1_start" not in marks:
            marks["phase1_start"] = elapsed
        if s.startswith("Faza 0+1 hotovo") and "phase1_end" not in marks:
            marks["phase1_end"] = elapsed
        if s.startswith("Faza 2A:") and "phase2a_start" not in marks:
            marks["phase2a_start"] = elapsed
        if s.startswith("Faza 2A hotovo") and "phase2a_end" not in marks:
            marks["phase2a_end"] = elapsed

    try:
        result = run_full_photometry_pipeline(
            masterstar_fits_path=og / "MASTERSTAR.fits",
            variable_targets_csv=og / "variable_targets.csv",
            masterstars_csv=og / "masterstars_full_match.csv",
            per_frame_csv_dir=pf_dir,
            detrended_aligned_dir=pf_dir,
            output_dir=phot,
            cfg=cfg,
            db=db,
            draft_id=draft_id,
            progress_cb=_prog,
        )
        elapsed = time.perf_counter() - t0
        print(f"ELAPSED_S {elapsed:.1f}", flush=True)
        p0 = marks.get("phase0_end", 0) - marks.get("phase0_start", 0)
        p1 = marks.get("phase1_end", 0) - marks.get("phase1_start", 0)
        p2 = marks.get("phase2a_end", elapsed) - marks.get("phase2a_start", 0)
        print(f"PHASE0_S {p0:.1f}", flush=True)
        print(f"PHASE1_S {p1:.1f}", flush=True)
        print(f"PHASE2A_S {p2:.1f}", flush=True)
        if isinstance(result, dict):
            print(f"ERROR {result.get('error')!r}", flush=True)
            print(f"ZERO_TARGETS {result.get('zero_targets')!r}", flush=True)
            for k in ("n_lc", "n_lightcurves", "n_targets", "n_active"):
                if k in result:
                    print(f"{k.upper()} {result.get(k)!r}", flush=True)
            p2a = result.get("phase2a") if isinstance(result.get("phase2a"), dict) else {}
            if p2a:
                print(f"P2A_N_LC {p2a.get('n_lc')!r}", flush=True)
            print(f"RESULT_KEYS {sorted(result.keys())}", flush=True)
            return 1 if result.get("error") else 0
        print(f"RESULT_TYPE {type(result)}", flush=True)
        return 0
    finally:
        try:
            logf.flush()
            logf.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
