"""Photon-transfer gain / read noise from a folder of raw FITS (command line; no production behaviour).

The same measurement as Calibration Library > "Measure gain and read noise from raw flats"
(``gain_ptc.ptc_frames_from_folder`` + ``gain_ptc.measure_from_files``), for a folder on disk. Nothing is
written to the database; the JSON report (``--json``) is the ``PtcResult.to_json`` that "Save to camera" stores.

Needs RAW frames (not masters) of ONE gain setting and binning, recognised by IMAGETYP:
  flats: pairs of consecutive flats of the same exposure, at several levels (EMVA: >= 9, best ~25 from 0.5 % to
         90 % of saturation); darks: two darks for every flat exposure (dark variance removed per level), the
         shortest dark pair also gives the read noise.
A folder holding only one exposure of flats (a set for a master flat) gives at most one level: run --inventory
first.

Usage (repo root):
  python dev/tools/ptc_from_folder.py D:/FLAT56 --inventory
  python dev/tools/ptc_from_folder.py D:/FLAT56 --saturation-adu 65532 --adc-bits 14 --json ptc.json
  python dev/tools/ptc_from_folder.py D:/FLAT56 --db <data_root>/vyvar.sqlite3 --camera-id 3
    (--db / --camera-id read SATURATE_ADU and ADC_BITS of the camera; the database is opened read-only)
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from astropy.io import fits  # noqa: E402

import gain_ptc  # noqa: E402


def _key(path: Path) -> tuple[float, Any, Any]:
    with fits.open(path, memmap=False) as h:
        hd = h[0].header
    exp = gain_ptc._hdr_num(hd, "EXPTIME", "EXPOSURE")
    return (round(float(exp or 0.0), 4), gain_ptc._hdr_num(hd, "GAIN"), gain_ptc._hdr_num(hd, "XBINNING") or 1)


def inventory(flats: list[Path], darks: list[Path]) -> dict[str, Any]:
    """Frames per exposure and the (gain setting, binning) sets - what the PTC can use."""
    fk = [_key(p) for p in flats]
    dk = [_key(p) for p in darks]
    f_exp = Counter(k[0] for k in fk)
    d_exp = Counter(k[0] for k in dk)
    settings = sorted({(k[1], k[2]) for k in fk + dk}, key=str)
    levels = sorted(e for e, n in f_exp.items() if n >= 2)
    problems = []
    if len(settings) > 1:
        problems.append(f"more than one (gain setting, binning): {settings} - measure_from_files refuses mixed sets")
    if len(levels) < 3:
        problems.append(f"{len(levels)} flat exposure(s) with a pair - not a PTC ladder (need >= 3, best ~25); "
                        "a set of flats of one exposure (for a master flat) gives at most one level")
    missing = [e for e in levels if d_exp.get(e, 0) < 2]
    if missing:
        problems.append(f"no dark pair at flat exposure(s) {missing} s: dark variance stays in the fit intercept")
    if not dk:
        problems.append("no darks: no dark level, no read noise")
    return {
        "n_flats": len(flats), "n_darks": len(darks),
        "flats_per_exposure": {f"{e:g}": n for e, n in sorted(f_exp.items())},
        "darks_per_exposure": {f"{e:g}": n for e, n in sorted(d_exp.items())},
        "gain_setting_binning": [list(s) for s in settings],
        "flat_levels_with_pair": len(levels),
        "problems": problems,
    }


def _camera_values(db_path: Path, camera_id: int) -> tuple[float | None, int | None]:
    """(SATURATE_ADU, ADC_BITS) of the camera as the Calibration Library page reads them (ADC_BITS through the
    camera profile, i.e. EQUIPMENTS, else the camera catalog); the database is opened read-only."""
    from camera_profile import profile_from_connection  # noqa: PLC0415

    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT SATURATE_ADU FROM EQUIPMENTS WHERE ID = ?;", (int(camera_id),)).fetchone()
        if row is None:
            raise SystemExit(f"EQUIPMENTS ID={camera_id} not found in {db_path}")
        adc = profile_from_connection(con, int(camera_id)).adc_bits
    finally:
        con.close()
    sat = float(row[0]) if row[0] is not None and math.isfinite(float(row[0])) and float(row[0]) > 0 else None
    return sat, adc


def run(folder: Path, *, saturation_adu: float, adc_bits: int | None,
        frames: tuple[list[Path], list[Path]] | None = None) -> tuple[gain_ptc.PtcResult, dict[str, Any]]:
    flats, darks = frames if frames is not None else gain_ptc.ptc_frames_from_folder(folder)
    res = gain_ptc.measure_from_files(flats, darks, saturation_adu=float(saturation_adu))
    out: dict[str, Any] = {"folder": str(folder), "ok": res.ok, "gain_file": res.gain, "gain_file_err": res.gain_err,
                           "adc_bits": adc_bits, "read_noise_e": res.read_noise_e, "n_levels": res.n_levels,
                           "r2": res.r2, "through_origin": res.through_origin, "gain_setting": res.gain_setting,
                           "binning": res.binning, "covariance": res.covariance, "messages": res.messages}
    if res.ok:
        out["gain_native"] = gain_ptc.native_gain(float(res.gain), adc_bits)
        out["gain_native_err"] = gain_ptc.native_gain(float(res.gain_err), adc_bits) if res.gain_err else None
    return res, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--inventory", action="store_true", help="only list the usable frames, do not measure")
    ap.add_argument("--saturation-adu", type=float, help="saturation of the raw files [file ADU]")
    ap.add_argument("--adc-bits", type=int, help="ADC bit depth (native gain = file gain x 2^(16 - bits))")
    ap.add_argument("--db", type=Path, help="vyvar.sqlite3 to read SATURATE_ADU / ADC_BITS from (read-only)")
    ap.add_argument("--camera-id", type=int, help="EQUIPMENTS.ID of the camera (with --db)")
    ap.add_argument("--json", type=Path, help="write the full report (PtcResult.to_json + summary) here")
    a = ap.parse_args(argv)

    flats, darks = gain_ptc.ptc_frames_from_folder(a.folder)
    inv = inventory(flats, darks)
    print(json.dumps(inv, indent=2))
    if a.inventory:
        return 0
    sat, adc = a.saturation_adu, a.adc_bits
    if a.db is not None:
        if a.camera_id is None:
            ap.error("--db needs --camera-id")
        db_sat, db_adc = _camera_values(a.db, a.camera_id)
        sat = sat if sat is not None else db_sat
        adc = adc if adc is not None else db_adc
    if sat is None:
        ap.error("saturation unknown: pass --saturation-adu (no assumed 16-bit level) or --db/--camera-id")
    if adc is None:
        print("WARNING: ADC_BITS unknown - native gain = file gain (container scale 1)", file=sys.stderr)
    res, out = run(a.folder, saturation_adu=float(sat), adc_bits=adc, frames=(flats, darks))
    print(json.dumps(out, indent=2, default=str))
    if a.json is not None:
        a.json.write_text(json.dumps({"summary": out, "inventory": inv, "ptc": json.loads(res.to_json())},
                                     indent=2, default=str), encoding="ascii")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
