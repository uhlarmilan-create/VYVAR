"""Catalog database identity for run provenance (CATALOG-PROVENANCE)."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB for head/tail fingerprint (full SHA over 50 GB impractical)


def cheap_file_fingerprint(path: Path) -> str:
    """Stable fingerprint without reading the whole file (head + tail + size)."""
    p = Path(path).expanduser().resolve()
    size = int(p.stat().st_size)
    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    with p.open("rb") as fh:
        h.update(fh.read(_CHUNK))
        if size > _CHUNK:
            fh.seek(max(0, size - _CHUNK))
            h.update(fh.read(_CHUNK))
    return h.hexdigest()


def _sqlite_row_count(db_path: Path, table: str) -> int | None:
    try:
        con = sqlite3.connect(str(db_path))
        try:
            row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row else None
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return None


def fingerprint_gaia_db(path: str | Path) -> dict[str, Any] | None:
    p = Path(str(path or "")).expanduser()
    if not p.is_file():
        return None
    st = p.stat()
    out: dict[str, Any] = {
        "kind": "gaia_dr3_sqlite",
        "path": str(p.resolve()),
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "fingerprint_sha256": cheap_file_fingerprint(p),
        "fingerprint_method": "sha256(size + first_1MiB + last_1MiB)",
        "row_count": _sqlite_row_count(p, "gaia_dr3"),
    }
    try:
        from database import get_gaia_db_max_g_mag  # noqa: PLC0415

        out["max_g_mag"] = float(get_gaia_db_max_g_mag(p))
    except Exception:  # noqa: BLE001
        out["max_g_mag"] = None
    return out


def fingerprint_vsx_db(path: str | Path) -> dict[str, Any] | None:
    p = Path(str(path or "")).expanduser()
    if not p.is_file():
        return None
    st = p.stat()
    return {
        "kind": "vsx_local_sqlite",
        "path": str(p.resolve()),
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "fingerprint_sha256": cheap_file_fingerprint(p),
        "fingerprint_method": "sha256(size + first_1MiB + last_1MiB)",
        "row_count": _sqlite_row_count(p, "vsx_data"),
    }


def build_catalog_provenance_block(cfg: Any) -> dict[str, Any]:
    gaia = fingerprint_gaia_db(getattr(cfg, "gaia_db_path", "") or "")
    vsx = fingerprint_vsx_db(getattr(cfg, "vsx_local_db_path", "") or "")
    return {"gaia_dr3": gaia, "vsx_local": vsx}


def catalog_fingerprints_equal(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or not b:
        return a == b
    keys = ("fingerprint_sha256", "size_bytes", "row_count", "max_g_mag")
    return all(a.get(k) == b.get(k) for k in keys)


def summarize_catalog_delta(
    expected: dict[str, Any] | None,
    actual: dict[str, Any] | None,
) -> list[str]:
    if expected is None and actual is None:
        return []
    if expected is None or actual is None:
        return ["catalog block missing on one side"]
    issues: list[str] = []
    for name in ("gaia_dr3", "vsx_local"):
        exp = (expected or {}).get(name)
        act = (actual or {}).get(name)
        if catalog_fingerprints_equal(exp, act):
            continue
        issues.append(f"{name}: input catalogue changed")
        if exp and act:
            for k in ("fingerprint_sha256", "size_bytes", "row_count", "max_g_mag", "mtime_utc"):
                if exp.get(k) != act.get(k):
                    issues.append(f"  {name}.{k}: anchor={exp.get(k)!r} run={act.get(k)!r}")
    return issues
