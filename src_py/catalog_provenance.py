"""Catalog database identity for run provenance (CATALOG-PROVENANCE)."""

from __future__ import annotations

import hashlib
import logging
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
    from catalog_db import connect_ro  # noqa: PLC0415

    try:
        con = connect_ro(db_path)
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
    # G05-55: read-only - provenance never writes the Gaia DB; an older DB without its coverage
    # record (registered by the first top-up or gaia_maintain.py --register) is "unregistered".
    try:
        from gaia_coverage import coverage_is_stored, coverage_summary  # noqa: PLC0415

        coverage = coverage_summary(p) if coverage_is_stored(p) else "unregistered"
    except Exception as exc:  # noqa: BLE001
        logging.warning("[PROVENANCE] Gaia coverage of %s not read: %s", p, exc)
        coverage = None
    # Row count from catalog_meta (written by the builder / gaia_maintain.py --register, kept
    # current by the online top-up); COUNT(*) over 200 million rows took ~25 s per call.
    # A DB never registered is counted once here.
    from catalog_db import read_catalog_meta  # noqa: PLC0415

    meta = read_catalog_meta(p).get("gaia_dr3") or {}
    n = meta.get("row_count")
    st = p.stat()
    return {
        "kind": "gaia_dr3_sqlite",
        "path": str(p.resolve()),
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "fingerprint_sha256": cheap_file_fingerprint(p),
        "fingerprint_method": "sha256(size + first_1MiB + last_1MiB)",
        "row_count": int(n) if n is not None else _sqlite_row_count(p, "gaia_dr3"),
        "row_count_source": "catalog_meta" if n is not None else "count",
        "built_utc": meta.get("built_utc"),
        "build_params": meta.get("params") or None,
        "coverage": coverage,
    }


def fingerprint_objects_table(path: str | Path, table: str) -> dict[str, Any] | None:
    """Identity of one table of the objects DB (``vsx_data`` or ``exoplanet_data``).

    Per table, so an update of the exoplanet table does not change the VSX identity. The content
    fingerprint is the one ``catalogs/objects_make.py`` wrote into ``catalog_meta``; a DB without
    that row is fingerprinted from its rows here (slower, same definition).
    """
    from catalog_db import connect_ro, read_catalog_meta, table_content_sha256  # noqa: PLC0415

    p = Path(str(path or "")).expanduser()
    if not p.is_file():
        return None
    n = _sqlite_row_count(p, table)
    if n is None:
        return None
    meta = read_catalog_meta(p).get(table) or {}
    sha = meta.get("content_sha256")
    if not sha or meta.get("row_count") != n:
        con = connect_ro(p)
        try:
            sha = table_content_sha256(con, table)
        finally:
            con.close()
    return {
        "kind": "objects_sqlite_table",
        "path": str(p.resolve()),
        "table": table,
        "row_count": n,
        "built_utc": meta.get("built_utc"),
        "source": meta.get("source"),
        "fingerprint_sha256": sha,
        "fingerprint_method": "sha256 of the table rows ordered by key (catalog_db.table_content_sha256)",
    }


def build_catalog_provenance_block(cfg: Any) -> dict[str, Any]:
    objects = getattr(cfg, "objects_db_path", "") or ""
    return {
        "gaia_dr3": fingerprint_gaia_db(getattr(cfg, "gaia_db_path", "") or ""),
        "vsx": fingerprint_objects_table(objects, "vsx_data"),
        "exoplanets": fingerprint_objects_table(objects, "exoplanet_data"),
    }


def _coverage_key(cov: Any) -> Any:
    """Coverage rows without their timestamps (an in-memory registration of a read-only DB is
    re-created on every read)."""
    if not isinstance(cov, list):
        return cov
    keep = ("ra_min", "ra_max", "dec_min", "dec_max", "g_max", "source")
    return [{k: r.get(k) for k in keep} for r in cov if isinstance(r, dict)]


def catalog_fingerprints_equal(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or not b:
        return a == b
    keys = ("fingerprint_sha256", "size_bytes", "row_count")  # size_bytes: Gaia only (None == None for tables)
    return all(a.get(k) == b.get(k) for k in keys) and _coverage_key(a.get("coverage")) == _coverage_key(
        b.get("coverage")
    )
