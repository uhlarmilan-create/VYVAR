# -*- coding: ascii -*-
"""Read-only freeze of live 516 comparison inputs for A2-COMPARE.

Does not write Archive. Does not import photometry_lightcurve.
Does not instantiate VyvarDatabase.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src_py") not in sys.path:
    sys.path.insert(0, str(REPO / "src_py"))
if str(REPO / "dev") not in sys.path:
    sys.path.insert(0, str(REPO / "dev"))

from xval_pythonphot.run_xval_a1 import (  # noqa: E402
    CHECK_CID,
    G4_EXPECT,
    TARGET_CID,
    _sha256_file,
    g4_live_516,
    list_psf_lc_ids,
)

SETUP = "NoFilter_60_2"
LIVE_PS = REPO / "Archive" / "Drafts" / "draft_000516" / "platesolve" / SETUP
LIVE_LC = LIVE_PS / "photometry" / "lightcurves"
LIVE_PHOT = LIVE_PS / "photometry"
LIVE_LIGHTS = REPO / "Archive" / "Drafts" / "draft_000516" / "detrended_aligned" / "lights" / SETUP
SNAP_QC = (
    REPO / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
    / "calibrated" / "lights" / "qc_metrics.csv"
)
LIVE_QC = REPO / "Archive" / "Drafts" / "draft_000516" / "calibrated" / "lights" / "qc_metrics.csv"
MS_PATH = LIVE_PS / "masterstars_full_match.csv"
KIT = Path(__file__).resolve().parent
CTX = REPO / "dev" / "results" / "context" / "session_20260907_epsfxval_a2" / "vyvar_reference"
ENS_IDS = [
    "1497771992240531712",
    "1499200223486564608",
    "1497974027502858240",
    "1497368849430107904",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _header(lines: list[str], body: str) -> str:
    block = "\n".join(lines)
    if not block.endswith("\n"):
        block += "\n"
    return block + body


def write_targets() -> list[str]:
    g4 = g4_live_516()
    digest = g4["csv"]["sha256"]
    lc_ids = list_psf_lc_ids(LIVE_LC)
    wanted = sorted(set(lc_ids) | {TARGET_CID, CHECK_CID} | set(ENS_IDS))
    ms = pd.read_csv(MS_PATH, dtype={"catalog_id": str})
    rows = []
    missing = []
    for cid in wanted:
        hit = ms[ms["catalog_id"].astype(str) == cid]
        if hit.empty:
            missing.append(cid)
            continue
        r = hit.iloc[0]
        role = []
        if cid == TARGET_CID:
            role.append("target")
        if cid == CHECK_CID:
            role.append("check")
        if cid in ENS_IDS:
            role.append("ensemble")
        if cid in lc_ids:
            role.append("psf_lc")
        rows.append({
            "catalog_id": cid,
            "ra": float(r["ra_deg"]),
            "dec": float(r["dec_deg"]),
            "role": "+".join(role) if role else "other",
        })
    if missing:
        raise SystemExit(f"targets missing from masterstars: {missing}")
    header = [
        f"# provenance_source={MS_PATH.as_posix()}",
        f"# sha256={digest}",
        f"# sha256_prefix={G4_EXPECT['csv']}",
        f"# extracted_utc={_now()}",
        f"# n={len(rows)}",
        "# columns=catalog_id,ra,dec (+ role diagnostic; A2-COMPARE keys on catalog_id)",
        "# set=PSF LC stars + pinned ensemble + check (same membership as A1/A1B)",
    ]
    df = pd.DataFrame(rows)
    text = df.to_csv(index=False)
    (KIT / "targets.csv").write_text(_header(header, text), encoding="ascii", newline="\n")
    return wanted


def freeze(wanted: list[str]) -> None:
    CTX.mkdir(parents=True, exist_ok=True)
    g4 = g4_live_516()
    if not g4["pass"]:
        raise SystemExit(f"G4 FAIL before freeze: {g4}")

    # qc_metrics: prefer snapshot (same file the Linux kit will read)
    qc_src = SNAP_QC if SNAP_QC.is_file() else LIVE_QC
    qc_sha = _sha256_file(qc_src)
    qc_body = qc_src.read_text(encoding="utf-8")
    qc_head = [
        f"# provenance_source={qc_src.as_posix()}",
        f"# sha256={qc_sha}",
        f"# extracted_utc={_now()}",
        "# used_by=Linux run_all.sh SEEING_FWHM = fwhm_px * WCS scale",
    ]
    (CTX / "qc_metrics.csv").write_text(_header(qc_head, qc_body), encoding="utf-8", newline="\n")

    # per-frame proc fluxes
    procs = sorted(LIVE_LIGHTS.glob("proc_*.csv"))
    recs = []
    for p in procs:
        df = pd.read_csv(p, dtype={"catalog_id": str})
        sub = df[df["catalog_id"].astype(str).isin(wanted)]
        for _, r in sub.iterrows():
            recs.append({
                "source_file": p.name,
                "fits": str(r.get("source_file", "")),
                "catalog_id": str(r["catalog_id"]),
                "psf_flux": r.get("psf_flux"),
                "psf_flux_err": r.get("psf_flux_err"),
                "psf_chi2": r.get("psf_chi2"),
                "psf_fit_ok": r.get("psf_fit_ok"),
                "x": r.get("x"),
                "y": r.get("y"),
            })
    flux = pd.DataFrame(recs)
    flux_path = CTX / "proc_psf_flux.csv"
    flux_head = [
        f"# provenance_source={LIVE_LIGHTS.as_posix()}/proc_*.csv",
        f"# n_proc_files={len(procs)}",
        f"# n_rows={len(flux)}",
        f"# live_masterstars_sha256={g4['csv']['sha256']}",
        f"# live_masterstars_prefix={G4_EXPECT['csv']}",
        f"# live_epsf_sha256={g4['epsf']['sha256']}",
        f"# live_epsf_prefix={G4_EXPECT['epsf']}",
        f"# extracted_utc={_now()}",
        "# read_only=draft_000516 live proc; Archive not written",
    ]
    flux_path.write_text(_header(flux_head, flux.to_csv(index=False)), encoding="utf-8", newline="\n")

    # target PSF LC + ensemble sidecar (ensemble members have no dedicated PSF LC)
    lc_dir = CTX / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    tgt_lc = LIVE_LC / f"lightcurve_{TARGET_CID}_psf.csv"
    raw = tgt_lc.read_text(encoding="utf-8")
    lc_head = [
        f"# freeze_provenance_source={tgt_lc.as_posix()}",
        f"# sha256={_sha256_file(tgt_lc)}",
        f"# live_epsf_prefix={G4_EXPECT['epsf']}",
        f"# extracted_utc={_now()}",
    ]
    (lc_dir / tgt_lc.name).write_text(_header(lc_head, raw), encoding="utf-8", newline="\n")

    ens_copied = []
    ens_missing = []
    for cid in ENS_IDS:
        src = LIVE_LC / f"lightcurve_{cid}_psf.csv"
        if src.is_file():
            (lc_dir / src.name).write_text(
                _header(
                    [f"# freeze_provenance_source={src.as_posix()}", f"# sha256={_sha256_file(src)}", f"# extracted_utc={_now()}"],
                    src.read_text(encoding="utf-8"),
                ),
                encoding="utf-8",
                newline="\n",
            )
            ens_copied.append(cid)
        else:
            ens_missing.append(cid)

    sidecar = {
        "target_cid": TARGET_CID,
        "check_cid": CHECK_CID,
        "ensemble_ids": ENS_IDS,
        "ensemble_source": "pinned",
        "ensemble_psf_lc_present": ens_copied,
        "ensemble_psf_lc_absent": ens_missing,
        "note": "ensemble fluxes for A2-COMPARE come from proc_psf_flux.csv (same as A1); sidecar ids/weights stay on the target LC header",
        "target_lc_header_gain_authority": "g_pt=0.637067 source=g_pt",
        "g4": g4,
        "extracted_utc": _now(),
    }
    # weights from pinned_ensembles without opening the app config if possible
    pin_csv = REPO / "dev" / "validation" / "pinned_ensembles.csv"
    weights: dict[str, float] = {}
    if pin_csv.is_file():
        pin = pd.read_csv(pin_csv, dtype=str)
        tcol = "target_catalog_id" if "target_catalog_id" in pin.columns else pin.columns[0]
        cidcol = "comp_catalog_id" if "comp_catalog_id" in pin.columns else "catalog_id"
        wcol = "comp_weight" if "comp_weight" in pin.columns else "weight"
        sub = pin[pin[tcol].astype(str) == TARGET_CID]
        for _, row in sub.iterrows():
            cid = str(row.get(cidcol, "")).strip()
            try:
                w = float(row.get(wcol))
            except (TypeError, ValueError):
                continue
            if cid:
                weights[cid] = w
    sidecar["ensemble_weights"] = weights
    sidecar["ensemble_weights_source"] = str(pin_csv.as_posix())

    (CTX / "ensemble_sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    (CTX / "PROVENANCE.md").write_text(
        "\n".join([
            "# VYVAR reference freeze for EPSF-XVAL-A2-COMPARE",
            "",
            f"extracted_utc={_now()}",
            f"live_ps={LIVE_PS.as_posix()}",
            f"masterstars_sha256={g4['csv']['sha256']}",
            f"masterstars_prefix={G4_EXPECT['csv']}",
            f"masterstar_fits_sha256={g4['fits']['sha256']}",
            f"masterstar_fits_prefix={G4_EXPECT['fits']}",
            f"epsf_sha256={g4['epsf']['sha256']}",
            f"epsf_prefix={G4_EXPECT['epsf']}",
            f"qc_source={qc_src.as_posix()}",
            f"qc_sha256={qc_sha}",
            "read_only=yes",
            "archive_written=no",
            "",
        ]),
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps({
        "n_targets": len(wanted),
        "n_proc": len(procs),
        "n_flux_rows": int(len(flux)),
        "ensemble_psf_lc_absent": ens_missing,
        "g4": g4["pass"],
        "ctx": str(CTX),
    }, indent=2))


def main() -> int:
    wanted = write_targets()
    freeze(wanted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
