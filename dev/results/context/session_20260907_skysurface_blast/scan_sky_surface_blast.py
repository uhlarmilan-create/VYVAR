# -*- coding: ascii -*-
"""SKY-SURFACE-BLAST-RADIUS-01: read-only export forensic scan, drafts 438-451.

Does not write Archive, does not open the DB read-write, does not run photometry.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\ASTRO\python\VYVAR")
OUT = REPO / "dev" / "results" / "context" / "session_20260907_skysurface_blast"
IDS = list(range(438, 452))
DRAFT_DIR_RE = re.compile(r"^draft_0*(\d+)$", re.I)


def _parse_config_json(text: str) -> dict:
    lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("//"):
            continue
        if "//" in raw:
            in_str = False
            out_ch: list[str] = []
            i = 0
            while i < len(raw):
                ch = raw[i]
                if ch == '"' and (i == 0 or raw[i - 1] != "\\"):
                    in_str = not in_str
                if not in_str and raw[i : i + 2] == "//":
                    break
                out_ch.append(ch)
                i += 1
            raw = "".join(out_ch).rstrip()
        if raw.strip():
            lines.append(raw)
    return json.loads("\n".join(lines))


def archive_and_db_paths() -> tuple[Path, Path]:
    cfg_path = REPO / "config.json"
    data = _parse_config_json(cfg_path.read_text(encoding="utf-8"))
    arch = Path(str(data.get("archive_root") or (REPO / "Archive")))
    dbp = Path(str(data.get("database_path") or (REPO / "vyvar.sqlite3")))
    return arch, dbp


def iso_mtime(p: Path) -> str:
    ts = p.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def draft_standard_dir(archive_root: Path, did: int) -> Path:
    return archive_root / "Drafts" / f"draft_{did:06d}"


def find_relocated_dirs(archive_root: Path) -> dict[int, list[str]]:
    found: dict[int, list[str]] = {i: [] for i in IDS}
    if not archive_root.is_dir():
        return found
    for child in archive_root.rglob("*"):
        if not child.is_dir():
            continue
        m = DRAFT_DIR_RE.match(child.name)
        if m:
            did = int(m.group(1))
            if did in found:
                found[did].append(str(child))
        mf = child / "draft_manifest.json"
        if mf.is_file():
            try:
                raw = json.loads(mf.read_text(encoding="utf-8"))
                did = int(raw.get("draft_id", -1))
                if did in found and str(child) not in found[did]:
                    found[did].append(str(child))
            except Exception:
                continue
    return found


def object_setup_dirs(draft_root: Path) -> list[dict]:
    out: list[dict] = []
    platesolve = draft_root / "platesolve"
    if platesolve.is_dir():
        for setup in sorted(p for p in platesolve.iterdir() if p.is_dir()):
            out.append({"kind": "platesolve_setup", "name": setup.name, "path": str(setup)})
    for top in ("Raw", "calibrated", "processed"):
        p = draft_root / top
        if p.is_dir():
            kids = [c.name for c in p.iterdir() if c.is_dir()]
            out.append({"kind": top, "name": top, "children": kids, "path": str(p)})
    return out


def parse_aavso_txt(path: Path) -> dict:
    target = None
    band = None
    jds: list[float] = []
    n_rows = 0
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            if s.startswith("#NAME=") or s.startswith("#STARID="):
                target = s.split("=", 1)[1].strip() or target
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 5:
            continue
        try:
            jd = float(parts[1])
        except ValueError:
            continue
        n_rows += 1
        jds.append(jd)
        if target is None:
            target = parts[0] or None
        if band is None:
            band = parts[4] or None
    return {
        "target": target,
        "band": band,
        "jd_first": jds[0] if jds else None,
        "jd_last": jds[-1] if jds else None,
        "n_rows": n_rows,
    }


def parse_varastro_txt(path: Path) -> dict:
    target = None
    band = None
    jds: list[float] = []
    n_rows = 0
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# VAR Name:"):
            rest = s[len("# VAR Name:") :].strip()
            target = rest.split("|", 1)[0].strip() or target
            continue
        if s.startswith("#   Filter:"):
            tok = s.split("|", 1)[0]
            band = tok.replace("#   Filter:", "").strip() or band
            continue
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if not parts:
            continue
        try:
            jd = float(parts[0])
        except ValueError:
            continue
        if jd < 2000000.0:
            continue
        n_rows += 1
        jds.append(jd)
    return {
        "target": target,
        "band": band,
        "jd_first": jds[0] if jds else None,
        "jd_last": jds[-1] if jds else None,
        "n_rows": n_rows,
    }


def inventory_reports(draft_id: int, setup: str, reports: Path) -> list[dict]:
    rows: list[dict] = []
    for fp in sorted(reports.rglob("*")):
        if not fp.is_file():
            continue
        rel = str(fp.relative_to(reports)).replace("\\", "/")
        rec: dict = {
            "draft_id": draft_id,
            "setup": setup,
            "file": rel,
            "abs_path": str(fp),
            "size": fp.stat().st_size,
            "mtime": iso_mtime(fp),
            "target": None,
            "band": None,
            "jd_first": None,
            "jd_last": None,
            "n_rows": None,
            "kind": "other",
        }
        low = rel.lower()
        if low.endswith(".txt") and "/aavso/" in f"/{low}" or (
            low.endswith(".txt") and low.startswith("aavso/")
        ):
            rec["kind"] = "aavso"
            rec.update(parse_aavso_txt(fp))
        elif low.endswith(".txt") and ("/varastro/" in f"/{low}" or low.startswith("varastro/")):
            rec["kind"] = "varastro"
            rec.update(parse_varastro_txt(fp))
        elif low.endswith(".png") and "varastro" in low:
            rec["kind"] = "varastro_png"
        rows.append(rec)
    return rows


def adhoc_glob(draft_root: Path) -> list[dict]:
    hits: list[dict] = []
    pats = ("*aavso*", "*webobs*", "*varastro*")
    seen: set[str] = set()
    for pat in pats:
        for fp in draft_root.rglob(pat):
            if not fp.is_file():
                continue
            key = str(fp).lower()
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                {
                    "path": str(fp),
                    "size": fp.stat().st_size,
                    "mtime": iso_mtime(fp),
                }
            )
    return hits


def db_readonly_lookup(db_path: Path) -> dict:
    out: dict = {
        "path": str(db_path),
        "exists": db_path.is_file(),
        "mode": "uri_ro",
        "ok": False,
        "error": None,
        "tables": [],
        "rows": {},
    }
    if not db_path.is_file():
        out["error"] = "database file absent"
        return out
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            out["tables"] = tables
            id_list = ",".join(str(i) for i in IDS)
            for table in tables:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
                col_u = {c.upper(): c for c in cols}
                if "DRAFT_ID" not in col_u:
                    continue
                pick = col_u["DRAFT_ID"]
                date_col = None
                for cand in ("CREATED_AT", "OBS_DATE", "DATE", "JD"):
                    if cand in col_u:
                        date_col = col_u[cand]
                        break
                if date_col:
                    q = (
                        f"SELECT {pick} AS draft_id, COUNT(*) AS n, "
                        f"MIN({date_col}) AS date_min, MAX({date_col}) AS date_max "
                        f"FROM {table} WHERE {pick} IN ({id_list}) GROUP BY {pick} "
                        f"ORDER BY {pick}"
                    )
                else:
                    q = (
                        f"SELECT {pick} AS draft_id, COUNT(*) AS n "
                        f"FROM {table} WHERE {pick} IN ({id_list}) GROUP BY {pick} "
                        f"ORDER BY {pick}"
                    )
                slim = [dict(r) for r in conn.execute(q)]
                if slim:
                    out["rows"][table] = slim
            out["ok"] = True
        finally:
            conn.close()
    except Exception as exc:
        out["error"] = str(exc)
    return out


def archive_wide_export_names(archive_root: Path) -> list[str]:
    hits: list[str] = []
    if not archive_root.is_dir():
        return hits
    for fp in archive_root.rglob("*"):
        if not fp.is_file():
            continue
        name = fp.name.lower()
        if any(tok in name for tok in ("aavso", "webobs", "varastro")):
            hits.append(str(fp))
    return hits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive_root, db_path = archive_and_db_paths()
    relocated = find_relocated_dirs(archive_root)
    db_info = db_readonly_lookup(db_path)
    drafts: list[dict] = []
    fingerprints: list[dict] = []
    adhoc_all: list[dict] = []
    reports_trees: list[dict] = []

    for did in IDS:
        std = draft_standard_dir(archive_root, did)
        extra = [p for p in relocated[did] if Path(p).resolve() != std.resolve()]
        exists = std.is_dir()
        db_date = None
        for table in ("OBS_QC_PROCESSING_RUN", "MASTER_SOURCES"):
            for row in db_info.get("rows", {}).get(table, []):
                if int(row.get("draft_id") or 0) == did:
                    db_date = row.get("date_min") or row.get("date_max")
                    if table == "OBS_QC_PROCESSING_RUN":
                        break
            if table == "OBS_QC_PROCESSING_RUN" and db_date:
                break
        rec = {
            "draft_id": did,
            "standard_path": str(std),
            "directory_exists": exists,
            "relocated_paths": extra,
            "object_setup_dirs": [],
            "db_date": db_date,
            "reports_dirs": [],
        }
        roots = []
        if exists:
            roots.append(std)
        for p in extra:
            rp = Path(p)
            if rp.is_dir() and rp.resolve() not in {r.resolve() for r in roots}:
                roots.append(rp)
        for root in roots:
            rec["object_setup_dirs"].extend(object_setup_dirs(root))
            platesolve = root / "platesolve"
            if platesolve.is_dir():
                for setup in sorted(p for p in platesolve.iterdir() if p.is_dir()):
                    reports = setup / "photometry" / "lightcurves_reports"
                    rec["reports_dirs"].append(
                        {"setup": setup.name, "exists": reports.is_dir(), "path": str(reports)}
                    )
                    if reports.is_dir():
                        inv = inventory_reports(did, setup.name, reports)
                        reports_trees.append(
                            {
                                "draft_id": did,
                                "setup": setup.name,
                                "n_files": len(inv),
                                "files": inv,
                            }
                        )
                        for item in inv:
                            fingerprints.append(
                                {
                                    "draft_id": did,
                                    "setup": setup.name,
                                    "file": item["file"],
                                    "target": item.get("target"),
                                    "band": item.get("band"),
                                    "JD_first": item.get("jd_first"),
                                    "JD_last": item.get("jd_last"),
                                    "n_rows": item.get("n_rows"),
                                    "mtime": item.get("mtime"),
                                }
                            )
            hits = adhoc_glob(root)
            for h in hits:
                adhoc_all.append({"draft_id": did, **h})
        drafts.append(rec)

    archive_hits = archive_wide_export_names(archive_root)
    n_txt = sum(1 for r in fingerprints if str(r.get("file", "")).lower().endswith(".txt"))
    reading = "R-S1" if n_txt == 0 and not fingerprints else "R-S2"

    inventory = {
        "archive_root": str(archive_root),
        "database_path": str(db_path),
        "ids": IDS,
        "drafts": drafts,
        "reports_trees": reports_trees,
        "adhoc": adhoc_all,
        "archive_wide_name_hits": archive_hits,
        "db": db_info,
    }
    summary = {
        "archive_root": str(archive_root),
        "n_ids": len(IDS),
        "n_dirs_present": sum(1 for d in drafts if d["directory_exists"] or d["relocated_paths"]),
        "n_dirs_absent": sum(
            1 for d in drafts if not d["directory_exists"] and not d["relocated_paths"]
        ),
        "n_fingerprint_rows": len(fingerprints),
        "n_export_txt": n_txt,
        "n_adhoc": len(adhoc_all),
        "n_archive_wide_name_hits": len(archive_hits),
        "db_ok": db_info.get("ok"),
        "db_error": db_info.get("error"),
        "reading": reading,
    }

    (OUT / "inventory.json").write_text(
        json.dumps(inventory, indent=2, default=str) + "\n", encoding="ascii"
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="ascii"
    )

    with (OUT / "fingerprint.csv").open("w", encoding="ascii", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "draft_id",
                "setup",
                "file",
                "target",
                "band",
                "JD_first",
                "JD_last",
                "n_rows",
                "mtime",
            ],
        )
        w.writeheader()
        for row in fingerprints:
            w.writerow(row)

    md: list[str] = []
    md.append("# Fingerprint table (438-451)")
    md.append("")
    md.append(
        "| draft id | setup | file | target | band | JD_first | JD_last | n_rows | mtime |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|")
    if fingerprints:
        for row in fingerprints:
            md.append(
                "| {draft_id} | {setup} | {file} | {target} | {band} | {JD_first} | {JD_last} | {n_rows} | {mtime} |".format(
                    **{k: ("" if row.get(k) is None else row.get(k)) for k in row}
                )
            )
    else:
        md.append("| (none) |  |  |  |  |  |  |  |  |")
    md.append("")
    md.append(f"Reading: **{reading}**. n_export_txt={n_txt}.")
    (OUT / "fingerprint.md").write_text("\n".join(md) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
