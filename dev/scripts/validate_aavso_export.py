#!/usr/bin/env python3
"""Validate VYVAR AAVSO Extended Format export files.

Usage:
    python scripts/validate_aavso_export.py --draft 321
    python scripts/validate_aavso_export.py --path "D:\\...\\lightcurves_reports\\aavso"

Exit code 0 if no errors (warnings OK); 1 if any errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# AAVSO Extended row column order (15 fields)
_COL_NAMES = (
    "STARID",
    "DATE",
    "MAG",
    "MAGERR",
    "FILTER",
    "TRANS",
    "MTYPE",
    "CNAME",
    "CMAG",
    "KNAME",
    "KMAG",
    "AMASS",
    "GROUP",
    "CHART",
    "NOTES",
)
_NCOLS = len(_COL_NAMES)

_VALID_FILTERS = frozenset(
    {
        "U",
        "B",
        "V",
        "R",
        "I",
        "RJ",
        "IJ",
        "Rc",
        "Ic",
        "J",
        "H",
        "K",
        "Y",
        "CV",
        "CR",
        "TB",
        "TG",
        "TR",
        "vis",
        "SZ",
        "SR",
        "SI",
        "SU",
        "UNKN",
    }
)
_VALID_TRANS = frozenset({"YES", "NO"})
_VALID_MTYPE = frozenset({"STD", "DIFF", "DIF", "ABS"})
_VALID_DATE_SCALES = frozenset({"BJD", "JD", "HJD"})
_VALID_OBSTYPE = frozenset({"CCD", "DSLR", "Visual"})

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_PROJECT_ROOT = _bootstrap.REPO_ROOT
_CONFIG_PATH = _PROJECT_ROOT / "config.json"


def _load_archive_root(config_path: Path | None) -> Path:
    cfg_file = config_path if config_path is not None else _CONFIG_PATH
    archive = _PROJECT_ROOT / "Archive"
    if cfg_file.is_file():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            archive = Path(str(data.get("archive_root", archive)))
        except (OSError, json.JSONDecodeError):
            pass
    return archive.resolve()


def resolve_draft_aavso_dirs(draft_id: int, *, config_path: Path | None = None) -> list[Path]:
    """Same layout as export_lightcurve_reports: {photometry}/lightcurves_reports/aavso/."""
    archive = _load_archive_root(config_path)
    draft_dir = (archive / "Drafts" / f"draft_{int(draft_id):06d}").resolve()
    if not draft_dir.is_dir():
        raise FileNotFoundError(f"Draft directory not found: {draft_dir}")

    aavso_dirs: list[Path] = []
    platesolve = draft_dir / "platesolve"
    if platesolve.is_dir():
        for phot_dir in sorted(platesolve.glob("*/photometry")):
            aavso = phot_dir / "lightcurves_reports" / "aavso"
            if aavso.is_dir():
                aavso_dirs.append(aavso.resolve())

    if not aavso_dirs:
        raise FileNotFoundError(
            f"No lightcurves_reports/aavso under {draft_dir / 'platesolve'}"
        )
    return aavso_dirs


def collect_txt_files(path_arg: Path) -> list[Path]:
    p = path_arg.resolve()
    if p.is_file() and p.suffix.lower() == ".txt":
        return [p]
    if p.is_dir():
        if p.name.lower() == "aavso":
            return sorted(p.glob("*.txt"))
        nested = p / "lightcurves_reports" / "aavso"
        if nested.is_dir():
            return sorted(nested.glob("*.txt"))
        return sorted(p.glob("*.txt"))
    raise FileNotFoundError(f"Path not found: {p}")


def parse_aavso_file(path: Path) -> tuple[dict[str, str], list[list[str]]]:
    """Parse AAVSO Extended file into header key/value dict and data rows."""
    header: dict[str, str] = {}
    rows: list[list[str]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if "=" in body:
                key, _, val = body.partition("=")
                header[key.strip().upper()] = val.strip()
            continue
        rows.append([c.strip() for c in line.split(",")])
    return header, rows


def check_header(header_dict: dict[str, str], filepath: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    name = filepath.name

    typ = header_dict.get("TYPE", "")
    if not typ:
        errors.append(f"HEADER: missing TYPE in {name}")
    elif typ != "Extended":
        errors.append(f"HEADER: TYPE must be 'Extended', got '{typ}' in {name}")

    obscode = header_dict.get("OBSCODE", "")
    if not obscode:
        warnings.append(
            f"HEADER: observer code not set - AAVSO submission requires an "
            f"observer code (config: aavso_observer_code) ({name})"
        )

    warn_msg = header_dict.get("WARNING", "")
    if warn_msg and "FILT unrecognized" in warn_msg:
        warnings.append(f"HEADER: {warn_msg} ({name})")

    delim = header_dict.get("DELIM", "")
    if not delim:
        errors.append(f"HEADER: missing DELIM in {name}")
    elif delim != ",":
        errors.append(f"HEADER: DELIM must be ',', got '{delim}' in {name}")

    date_scale = header_dict.get("DATE", "")
    if not date_scale:
        errors.append(f"HEADER: missing DATE in {name}")
    elif date_scale.upper() not in _VALID_DATE_SCALES:
        errors.append(
            f"HEADER: DATE must be one of {sorted(_VALID_DATE_SCALES)}, got '{date_scale}' in {name}"
        )

    obstype = header_dict.get("OBSTYPE", "")
    if not obstype:
        errors.append(f"HEADER: missing OBSTYPE in {name}")
    elif obstype not in _VALID_OBSTYPE:
        errors.append(
            f"HEADER: OBSTYPE must be one of {sorted(_VALID_OBSTYPE)}, got '{obstype}' in {name}"
        )

    return errors, warnings


def _parse_float(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    if not (v == v and abs(v) != float("inf")):  # finite
        return None
    return v


class _IssueReporter:
    """Cap repeated issue types per file (max 5 each)."""

    def __init__(self, limit: int = 5) -> None:
        self.limit = limit
        self._counts: dict[str, int] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _can_emit(self, kind: str, issue_type: str) -> bool:
        key = f"{kind}:{issue_type}"
        n = self._counts.get(key, 0)
        if n >= self.limit:
            return False
        self._counts[key] = n + 1
        return True

    def error(self, issue_type: str, msg: str) -> None:
        if self._can_emit("ERROR", issue_type):
            self.errors.append(msg)

    def warning(self, issue_type: str, msg: str) -> None:
        if self._can_emit("WARNING", issue_type):
            self.warnings.append(msg)


def check_rows(rows: list[list[str]], filepath: Path) -> tuple[list[str], list[str]]:
    rep = _IssueReporter(limit=5)
    fname = filepath.name

    for i, row in enumerate(rows, start=1):
        if len(row) != _NCOLS:
            rep.error(
                "column_count",
                f"ERROR row {i}: expected {_NCOLS} columns, got {len(row)} in {fname}",
            )
            continue

        starid, date_s, mag_s, magerr_s, filt, trans, mtype = (
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
        )
        cname = row[7]
        amass_s = row[11]

        if not starid:
            rep.error("starid_empty", f"ERROR row {i}: STARID must not be empty in {fname}")
        elif len(starid) > 30:
            rep.warning(
                "starid_long",
                f"WARNING row {i}: STARID longer than 30 chars ({len(starid)}) in {fname}",
            )

        jd = _parse_float(date_s)
        if jd is None or jd <= 2400000.0:
            rep.error(
                "date_invalid",
                f"ERROR row {i}: DATE must be float > 2400000.0, got '{date_s}' in {fname}",
            )

        mag = _parse_float(mag_s)
        if mag is None or not (1.0 <= mag <= 25.0):
            rep.error(
                "mag_range",
                f"ERROR row {i}: MAG must be float in 1.0-25.0, got '{mag_s}' in {fname}",
            )

        if magerr_s.strip().lower() == "na":
            pass
        else:
            magerr = _parse_float(magerr_s)
            if magerr is None or magerr <= 0:
                rep.error(
                    "magerr_invalid",
                    f"ERROR row {i}: MAGERR must be float > 0 or 'na', got '{magerr_s}' in {fname}",
                )

        if filt not in _VALID_FILTERS:
            rep.warning(
                "filter_unknown",
                f"WARNING row {i}: FILTER '{filt}' not in known AAVSO filter list in {fname}",
            )
        elif filt == "UNKN":
            rep.warning(
                "filter_unrecognized",
                f"WARNING row {i}: FILTER=UNKN - unrecognized band; review before submit in {fname}",
            )

        if trans not in _VALID_TRANS:
            rep.error(
                "trans_invalid",
                f"ERROR row {i}: TRANS must be YES or NO, got '{trans}' in {fname}",
            )

        if mtype not in _VALID_MTYPE:
            rep.error(
                "mtype_invalid",
                f"ERROR row {i}: MTYPE must be STD, DIFF, or ABS, got '{mtype}' in {fname}",
            )

        if mtype in ("DIFF", "DIF"):
            rep.warning(
                "mtype_diff_rare",
                f"WARNING row {i}: MTYPE={mtype} is rare for VYVAR standardized exports; "
                f"expect STD in {fname}",
            )

        if not cname:
            rep.error("cname_empty", f"ERROR row {i}: CNAME must not be empty in {fname}")

        if amass_s.strip().lower() == "na":
            pass
        else:
            am = _parse_float(amass_s)
            if am is None or am <= 0:
                rep.error(
                    "amass_invalid",
                    f"ERROR row {i}: AMASS must be float > 0 or 'na', got '{amass_s}' in {fname}",
                )

    return rep.errors, rep.warnings


def validate_file(path: Path) -> tuple[int, int, int, list[str], list[str]]:
    header, rows = parse_aavso_file(path)
    header_errors, header_warnings = check_header(header, path)
    row_errors, row_warnings = check_rows(rows, path)
    errors = header_errors + row_errors
    warnings = header_warnings + row_warnings
    return len(rows), len(errors), len(warnings), errors, warnings


def _print_block(lines: list[str]) -> None:
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "backslashreplace").decode("ascii"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VYVAR AAVSO Extended export files")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--draft", type=int, help="Draft ID (e.g. 321 -> draft_000321)")
    g.add_argument("--path", type=Path, help="Path to aavso/ dir or single .txt file")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.json for archive_root (default: project config.json)",
    )
    args = parser.parse_args()

    files: list[Path] = []
    label = ""

    try:
        if args.draft is not None:
            draft_id = int(args.draft)
            label = f"draft_{draft_id:06d}"
            for aavso_dir in resolve_draft_aavso_dirs(draft_id, config_path=args.config):
                files.extend(collect_txt_files(aavso_dir))
            files = sorted(set(files))
        else:
            assert args.path is not None
            files = collect_txt_files(args.path)
            label = str(args.path.resolve())
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    if not files:
        print("No .txt files found.")
        return 1

    total_errors = 0
    total_warnings = 0
    files_clean = 0
    files_with_errors = 0

    for fpath in files:
        n_rows, n_err, n_warn, errs, warns = validate_file(fpath)
        total_errors += n_err
        total_warnings += n_warn
        if n_err:
            files_with_errors += 1
        else:
            files_clean += 1

        _print_block(["-" * 41])
        _print_block([f"FILE: {fpath.name}"])
        _print_block([f"  Rows parsed: {n_rows}"])

        header_only = [e for e in errs if e.startswith("HEADER:")]
        row_errs = [e for e in errs if not e.startswith("HEADER:")]

        if header_only:
            for e in header_only:
                _print_block([f"  [X] {e.replace('HEADER: ', '')}"])
        else:
            _print_block(["  [OK] Header OK"])

        for w in warns:
            _print_block([f"  !  {w}"])
        for e in row_errs:
            _print_block([f"  [X] {e}"])

        if n_err == 0 and n_warn == 0:
            _print_block(["  RESULT: clean"])
        else:
            _print_block([f"  RESULT: {n_err} errors, {n_warn} warnings"])

    _print_block(["-" * 41])
    _print_block(["", "=" * 39])
    draft_label = label if args.draft is not None else "custom path"
    _print_block([f"AAVSO VALIDATION SUMMARY - {draft_label}"])
    _print_block([f"  Files checked     : {len(files)}"])
    _print_block([f"  Files clean       : {files_clean}"])
    _print_block([f"  Files with errors : {files_with_errors}"])
    _print_block([f"  Total errors      : {total_errors}"])
    _print_block([f"  Total warnings    : {total_warnings}"])
    if total_errors == 0:
        _print_block(["  EXIT: 0  (warnings only - review before submission)"])
    else:
        _print_block(["  EXIT: 1  (errors found - fix before submission)"])
    _print_block(["=" * 39])

    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
