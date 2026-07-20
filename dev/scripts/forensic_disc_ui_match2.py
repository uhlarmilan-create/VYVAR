#!/usr/bin/env python3
"""F3 discriminator: NightRun on main with UI-parity match radius (2.0\") + config SysRem."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from night_run import NightRunParams, run_night_pipeline

OUT = _ROOT / "tmp" / "headless_forensics"
SETUP = "NoFilter_60_2"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # UI parity: app.py RUN VYVAR passes cat_match_arc=2.0; SysRem from config.json (=False).
    params = NightRunParams(
        source_dir=Path(r"D:\BO_CVn"),
        equipment_id=1,
        telescope_id=1,
        config_path=None,
        sysrem_enabled=False,  # honor config.json / UI (do not force True)
        sysrem_n_iter=3,
        catalog_match_max_sep_arcsec=2.0,  # UI hardcoded value
        progress_cb=lambda msg: print(msg, flush=True),
    )
    print(
        f"discriminator start utc={datetime.now(timezone.utc).isoformat()} "
        f"match_sep={params.catalog_match_max_sep_arcsec} sysrem={params.sysrem_enabled}",
        flush=True,
    )
    result = run_night_pipeline(params)
    draft_id = int(result.draft_id) if result.draft_id is not None else None
    payload = {
        "success": bool(result.success),
        "draft_id": draft_id,
        "errors": list(result.errors),
        "n_lightcurves": result.n_lightcurves,
        "n_frames": result.n_frames,
        "elapsed_phases": dict(result.phase_timings),
        "params": {
            "catalog_match_max_sep_arcsec": params.catalog_match_max_sep_arcsec,
            "sysrem_enabled": params.sysrem_enabled,
        },
    }
    if draft_id is not None:
        from config import AppConfig
        import pandas as pd

        cfg = AppConfig()
        ms_path = (
            Path(cfg.archive_root)
            / "Drafts"
            / f"draft_{draft_id:06d}"
            / "platesolve"
            / SETUP
            / "masterstars_full_match.csv"
        )
        meta_path = ms_path.parent / "photometry" / "pipeline_meta.json"
        if ms_path.is_file():
            ms = pd.read_csv(ms_path, dtype={"catalog_id": str})
            matched = int(ms["catalog_id"].fillna("").astype(str).str.strip().ne("").sum())
            payload["census"] = {
                "n_ms": int(len(ms)),
                "matched": matched,
                "unmatched": int(len(ms) - matched),
                "vsx_true": int(
                    pd.to_numeric(ms.get("vsx_known_variable", 0), errors="coerce")
                    .fillna(0)
                    .astype(bool)
                    .sum()
                )
                if "vsx_known_variable" in ms.columns
                else None,
            }
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            payload["identity_qa"] = {
                k: meta.get(k)
                for k in (
                    "matched_world2pix_identity_n",
                    "matched_world2pix_identity_p95_px",
                    "matched_world2pix_identity_p99_px",
                )
            }
            payload["provenance"] = {
                k: (meta.get("provenance") or {}).get(k)
                for k in ("git_hash", "git_dirty", "entry_point")
            }
    outp = OUT / "discriminator_ui_match2_report.json"
    outp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {outp}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
