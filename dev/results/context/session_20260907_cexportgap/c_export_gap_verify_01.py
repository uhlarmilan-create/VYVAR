# -*- coding: ascii -*-
"""C-EXPORT-GAP-VERIFY-01: headless night_run export measurement on sandbox 516.

MEASUREMENT ONLY. Writes under tmp/ and the session evidence dir.
Never writes Archive/Drafts/draft_000516.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev"))
sys.path.insert(0, str(ROOT / "dev" / "scripts"))

from session_baseline_check import (  # noqa: E402
    DRAFT_ID,
    SETUP,
    SNAPSHOT_NAME,
    _copy_frozen_anchor_inputs,
)

EVIDENCE = ROOT / "dev" / "results" / "context" / "session_20260907_cexportgap"
WORK = ROOT / "tmp" / "session_20260907_cexportgap" / "sandbox"
LIVE_PS = ROOT / "Archive" / "Drafts" / "draft_000516" / "platesolve" / SETUP
LIVE_SHA_PREFIX = {
    "csv": "bfa24039",
    "fits": "13e77cf8",
    "epsf": "172f9540",
}
# Documented night_run CLI ids from snapshot draft_manifest.json rig (wide).
CLI_CAMERA = "1"
CLI_TELESCOPE = "1"
CLI_SITE = "2"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def g4_live() -> dict:
    out = {
        "csv": sha256_file(LIVE_PS / "masterstars_full_match.csv"),
        "fits": sha256_file(LIVE_PS / "MASTERSTAR.fits"),
        "epsf": sha256_file(LIVE_PS / "masterstar_epsf.fits"),
    }
    out["verdict"] = (
        out["csv"].startswith(LIVE_SHA_PREFIX["csv"])
        and out["fits"].startswith(LIVE_SHA_PREFIX["fits"])
        and out["epsf"].startswith(LIVE_SHA_PREFIX["epsf"])
    )
    return out


def aperture_lc_cids(lc_dir: Path) -> list[str]:
    cids: list[str] = []
    if not lc_dir.is_dir():
        return cids
    for p in sorted(lc_dir.glob("lightcurve_*.csv")):
        name = p.name
        if name.endswith("_psf.csv") or name.endswith("_adaptive.csv"):
            continue
        m = re.match(r"lightcurve_(\d+)\.csv$", name)
        if m:
            cids.append(m.group(1))
    return cids


def _cid_in_name_or_text(cid: str, path: Path) -> bool:
    if cid in path.name:
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return cid in text


def inventory(phot: Path, log_text: str) -> dict:
    lc_dir = phot / "lightcurves"
    aavso_dir = phot / "lightcurves_reports" / "aavso"
    vara_dir = phot / "lightcurves_reports" / "varastro"
    at_path = phot / "active_targets.csv"
    cids = aperture_lc_cids(lc_dir)
    aavso_files = sorted(aavso_dir.glob("*")) if aavso_dir.is_dir() else []
    vara_files = sorted(vara_dir.glob("*")) if vara_dir.is_dir() else []
    aavso_files = [p for p in aavso_files if p.is_file()]
    vara_files = [p for p in vara_files if p.is_file()]

    at_cids: list[str] = []
    if at_path.is_file():
        import pandas as pd

        at = pd.read_csv(at_path, low_memory=False)
        if "catalog_id" in at.columns:
            at_cids = [
                str(x).strip()
                for x in at["catalog_id"].tolist()
                if str(x).strip() and str(x).strip().lower() != "nan"
            ]

    skip_re = re.compile(
        r"\[EXPORT\] skip (\S+) aperture: no LC CSV",
    )
    skipped = skip_re.findall(log_text)
    summary_re = re.compile(
        r"\[EXPORT\] lightcurves_reports: (\d+) targets exported, (\d+) skipped.*"
    )
    summaries = summary_re.findall(log_text)
    fail_lines = [
        ln
        for ln in log_text.splitlines()
        if "[EXPORT]" in ln
        and (
            "failure" in ln.lower()
            or "failed" in ln.lower()
            or "error" in ln.lower()
            or "refused" in ln.lower()
            or "init failed" in ln.lower()
        )
    ]
    export_lines = [ln for ln in log_text.splitlines() if "[EXPORT]" in ln]

    rows = []
    missing_export = []
    for cid in cids:
        has_a = any(_cid_in_name_or_text(cid, p) for p in aavso_files)
        has_v = any(_cid_in_name_or_text(cid, p) for p in vara_files)
        # Named VSX files (BO_CVn) may not contain the Gaia id in the filename;
        # search file body already covers that. If still missing, try vsx-less
        # count later via n_aavso vs n_lc.
        rec = {
            "catalog_id": cid,
            "lc": True,
            "aavso": has_a,
            "varastro": has_v,
            "logged_skip": cid in skipped,
        }
        rows.append(rec)
        if not (has_a and has_v):
            missing_export.append(rec)

    skip_without_lc = [s for s in skipped if s not in cids]
    skip_with_lc = [s for s in skipped if s in cids]

    band_obs = None
    sample_aavso = None
    if aavso_files:
        sample_aavso = str(aavso_files[0].name)
        text = aavso_files[0].read_text(encoding="utf-8", errors="replace")
        data_line = ""
        for ln in text.splitlines():
            if ln and not ln.startswith("#"):
                data_line = ln
                break
        parts = data_line.split(",")
        band_obs = {
            "file": aavso_files[0].name,
            "first_data_line_preview": data_line[:180],
            "field_index_4_band": parts[4].strip() if len(parts) > 4 else None,
            "header_band_hash": None,
        }
        for ln in text.splitlines():
            if ln.startswith("#") and "BAND" in ln.upper():
                band_obs["header_band_hash"] = ln[:160]
                break

    n_lc = len(cids)
    n_aavso = len(aavso_files)
    n_vara = len(vara_files)
    # H1: every aperture LC target has both export products, minus logged
    # no-LC-CSV skips (those should not be in cids).
    h1 = (n_lc > 0) and (not missing_export) and (n_aavso >= n_lc) and (n_vara >= n_lc)
    return {
        "n_lc": n_lc,
        "n_aavso": n_aavso,
        "n_varastro": n_vara,
        "n_active_targets": len(at_cids),
        "n_logged_skip": len(skipped),
        "skip_without_lc": skip_without_lc,
        "skip_with_lc": skip_with_lc,
        "n_missing_export_given_lc": len(missing_export),
        "missing_export": missing_export,
        "per_target": rows,
        "summary_matches": summaries,
        "fail_lines": fail_lines,
        "export_lines": export_lines,
        "band_obs": band_obs,
        "sample_aavso": sample_aavso,
        "h1_holds": h1,
        "aavso_names": [p.name for p in aavso_files],
        "varastro_names": [p.name for p in vara_files],
        "lc_cids": cids,
    }


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    WORK.parent.mkdir(parents=True, exist_ok=True)
    live_ok = LIVE_PS.is_dir()
    assert live_ok, f"live 516 platesolve missing: {LIVE_PS}"

    g4_before = g4_live()
    (EVIDENCE / "g4_before.json").write_text(
        json.dumps(g4_before, indent=2) + "\n", encoding="ascii"
    )

    os.environ["VYVAR_P1_FORCE"] = "1"

    from config import AppConfig
    from night_run import parse_night_run_cli, resolve_night_run_cli_ids, run_night_photometry
    from pipeline import AstroPipeline
    from tools.reference_seed import seed_reference_observatory

    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.save_lightcurve_png = False
    cfg.per_frame_saturation_enabled = True

    snapshot = Path(cfg.archive_root) / "Drafts" / SNAPSHOT_NAME
    if not snapshot.is_dir():
        raise SystemExit(f"missing snapshot {snapshot}")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    ps, lights = _copy_frozen_anchor_inputs(snapshot, WORK)
    live_draft = (ROOT / "Archive" / "Drafts" / "draft_000516").resolve()
    if live_draft == WORK.resolve() or live_draft in WORK.resolve().parents:
        raise SystemExit("REFUSE: work_root resolved under live draft_000516")
    if live_draft == ps.resolve() or live_draft in ps.resolve().parents:
        raise SystemExit("REFUSE: platesolve dest resolved under live draft_000516")

    cli_argv = [
        "--source",
        str(WORK),
        "--camera",
        CLI_CAMERA,
        "--telescope",
        CLI_TELESCOPE,
        "--site",
        CLI_SITE,
        "--draft-dir",
        str(WORK),
    ]
    args = parse_night_run_cli(cli_argv)
    eq, tel, loc, missing = resolve_night_run_cli_ids(
        equipment_id=args.equipment_id,
        telescope_id=args.telescope_id,
        location_id=args.location_id,
        draft_dir=args.draft_dir,
        cfg=cfg,
    )
    if missing:
        raise SystemExit(f"CLI ids missing: {missing}")

    documented_cli = (
        f"python src_py/night_run.py --source {WORK} --camera {CLI_CAMERA} "
        f"--telescope {CLI_TELESCOPE} --site {CLI_SITE} --draft-dir {WORK}"
    )

    log_path = EVIDENCE / "night_run_photometry.log"
    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    root_log.addHandler(sh)

    logging.info("C-EXPORT-GAP-VERIFY-01 start")
    logging.info("documented_cli=%s", documented_cli)
    logging.info("parsed eq=%s tel=%s loc=%s", eq, tel, loc)
    logging.info("work_root=%s", WORK)
    logging.info("snapshot=%s", snapshot)
    logging.info(
        "NOTE: parse_night_run_cli used; main()/run_night_pipeline NOT called "
        "(full CLI imports from --source into archive_root). Measuring "
        "run_night_photometry existing_draft=True on G2 sandbox copy."
    )

    seed_reference_observatory(AstroPipeline(cfg).db)
    pipeline = AstroPipeline(cfg)

    t0 = time.time()
    phot = run_night_photometry(
        cfg=cfg,
        pipeline=pipeline,
        draft_id=int(DRAFT_ID),
        draft_dir_override=WORK,
        write_pdfs=False,
        existing_draft=True,
        epsf=False,
    )
    elapsed = time.time() - t0
    logging.info("run_night_photometry elapsed_s=%.1f errors=%s", elapsed, phot.get("errors"))

    phot_dir = Path(ps) / "photometry"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    inv = inventory(phot_dir, log_text)
    inv["elapsed_s"] = elapsed
    inv["phot_errors"] = list(phot.get("errors") or [])
    inv["documented_cli"] = documented_cli
    inv["measured_entry"] = (
        "run_night_photometry(existing_draft=True, draft_dir_override=sandbox, "
        "write_pdfs=False, epsf=False)"
    )
    inv["cli_ids"] = {"equipment_id": eq, "telescope_id": tel, "location_id": loc}
    inv["work_root"] = str(WORK)
    inv["phot_dir"] = str(phot_dir)
    inv["setup"] = SETUP
    inv["draft_id"] = int(DRAFT_ID)
    inv["snapshot"] = SNAPSHOT_NAME

    g4_after = g4_live()
    inv["g4_before"] = g4_before
    inv["g4_after"] = g4_after
    inv["g4_pass"] = bool(g4_before["verdict"] and g4_after["verdict"])
    inv["live_516_written"] = False

    (EVIDENCE / "inventory.json").write_text(
        json.dumps(inv, indent=2) + "\n", encoding="ascii"
    )
    (EVIDENCE / "g4_after.json").write_text(
        json.dumps(g4_after, indent=2) + "\n", encoding="ascii"
    )

    reports = phot_dir / "lightcurves_reports"
    ev_rep = EVIDENCE / "sample_exports"
    ev_rep.mkdir(parents=True, exist_ok=True)
    a_src = reports / "aavso"
    v_src = reports / "varastro"
    if a_src.is_dir():
        files = sorted(p for p in a_src.iterdir() if p.is_file())
        if files:
            shutil.copy2(files[0], ev_rep / f"aavso_{files[0].name}")
            # Prefer BO_CVn if present (NoFilter named target).
            for p in files:
                if p.name.startswith("BO_"):
                    shutil.copy2(p, ev_rep / f"aavso_{p.name}")
                    break
    if v_src.is_dir():
        files = sorted(p for p in v_src.iterdir() if p.is_file())
        if files:
            shutil.copy2(files[0], ev_rep / f"varastro_{files[0].name}")
            for p in files:
                if p.name.startswith("BO_"):
                    shutil.copy2(p, ev_rep / f"varastro_{p.name}")
                    break

    print("INVENTORY_SUMMARY")
    print(json.dumps({
        "n_lc": inv["n_lc"],
        "n_aavso": inv["n_aavso"],
        "n_varastro": inv["n_varastro"],
        "n_missing_export_given_lc": inv["n_missing_export_given_lc"],
        "h1_holds": inv["h1_holds"],
        "summaries": inv["summary_matches"],
        "g4_pass": inv["g4_pass"],
        "elapsed_s": elapsed,
    }, indent=2))
    return 0 if not inv["phot_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
