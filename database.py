"""SQLite database layer for the VYVAR project."""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astropy.io import fits

from infolog import log_event
from utils import normalize_telescope_focal_mm_for_plate_scale

_GAIA_INDEX_CHECK_DONE = False

# Resolved DB path -> MAX(g_mag) from ``gaia_dr3`` (discovered at runtime).
_GAIA_DB_GMAG_MAX_CACHE: dict[str, float] = {}


def get_gaia_db_max_g_mag(db_path: str | Path) -> float:
    """Return the maximum ``g_mag`` stored in this Gaia SQLite (``SELECT MAX(g_mag) FROM gaia_dr3``).

    Cached per resolved path. If the table is empty or the query fails, returns ``0.0`` (caller should treat as
    „no photometric depth“ / empty DB).
    """
    p = Path(db_path).expanduser().resolve()
    key = str(p)
    if key in _GAIA_DB_GMAG_MAX_CACHE:
        return float(_GAIA_DB_GMAG_MAX_CACHE[key])
    if not p.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {p}")
    out = 0.0
    con = sqlite3.connect(str(p))
    try:
        row = con.execute("SELECT MAX(g_mag) AS m FROM gaia_dr3 WHERE g_mag IS NOT NULL").fetchone()
        if row is not None and row[0] is not None:
            try:
                v = float(row[0])
                if math.isfinite(v) and v > 0:
                    out = float(v)
            except (TypeError, ValueError):
                out = 0.0
    except Exception:  # noqa: BLE001
        out = 0.0
        try:
            log_event(f"GAIA DB: MAX(g_mag) sa nepodarilo načítať z {p.name} — predpokladám 0.0")
        except Exception:  # noqa: BLE001
            pass
    finally:
        con.close()
    _GAIA_DB_GMAG_MAX_CACHE[key] = out
    try:
        if out > 0:
            log_event(f"GAIA DB: MAX(g_mag) v gaia_dr3 = {out:.3f} ({p.name})")
        else:
            log_event(f"GAIA DB: MAX(g_mag) = 0 alebo prázdna tabuľka ({p.name})")
    except Exception:  # noqa: BLE001
        pass
    return out


def query_local_gaia(
    db_path: str | Path,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    mag_limit: float | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Query local Gaia DR3 SQLite for a rectangular sky window (ICRS deg).

    The Gaia DB schema can evolve; this function auto-detects optional columns and returns them when present.
    """
    p = Path(db_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {p}")
    # Default mag cap: keep SQL row counts low (target ~300–500 stars per cone on full DB).
    if mag_limit is None:
        mag_limit = 11.5
    try:
        ml = float(mag_limit)
    except (TypeError, ValueError):
        ml = float("nan")
    if not math.isfinite(ml) or ml <= 0:
        ml = 11.5
    _gmax_db = get_gaia_db_max_g_mag(p)
    if _gmax_db > 0.0 and ml > float(_gmax_db):
        try:
            log_event(
                f"GAIA SQL: mag_limit {float(ml):.2f} > MAX(g_mag) v DB ({float(_gmax_db):.3f}) — orezávam."
            )
        except Exception:  # noqa: BLE001
            pass
        ml = float(_gmax_db)
    mag_limit = float(ml)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    global _GAIA_INDEX_CHECK_DONE
    try:
        conn.execute("PRAGMA automatic_index = ON;")
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ra ON gaia_dr3 (ra);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dec ON gaia_dr3 (dec);")
            conn.commit()
        except Exception:  # noqa: BLE001
            pass

        # Discover optional columns for forward-compatible queries.
        cur_cols = conn.execute("PRAGMA table_info('gaia_dr3');")
        cols = {str(r[1]).strip().lower() for r in cur_cols.fetchall()}
        if not _GAIA_INDEX_CHECK_DONE:
            try:
                idx_rows = conn.execute("PRAGMA index_list('gaia_dr3');").fetchall()
                idx_names = [str(r[1]) for r in idx_rows if len(r) > 1]
                has_ra_dec_idx = False
                for nm in idx_names:
                    row_sql = conn.execute(
                        "SELECT sql FROM sqlite_master WHERE type='index' AND name=? LIMIT 1;",
                        (nm,),
                    ).fetchone()
                    sql_txt = str(row_sql[0]).lower() if row_sql is not None and row_sql[0] is not None else ""
                    if "ra" in sql_txt and "dec" in sql_txt:
                        has_ra_dec_idx = True
                        break
                    nml = nm.lower()
                    if "ra" in nml and "dec" in nml:
                        has_ra_dec_idx = True
                        break
                if not has_ra_dec_idx:
                    log_event(
                        "GAIA DB: upozornenie — nenašiel sa zjavný index na (ra, dec); "
                        "dotazy môžu byť pomalé."
                    )
            except Exception:  # noqa: BLE001
                pass
            _GAIA_INDEX_CHECK_DONE = True
        base_cols = ["source_id", "ra", "dec", "g_mag", "bp_mag", "rp_mag", "bp_rp", "var_flag"]
        opt_cols: list[str] = []
        for c in ("g_flux_error_rel", "non_single_star", "phot_variable_flag"):
            if c in cols:
                opt_cols.append(c)
        # Backward/alternate schema: some DBs carry a string catalog id.
        if "catalog_id" in cols:
            opt_cols.append("catalog_id")
        sel = ", ".join(base_cols + opt_cols)
        # Always include g_mag constraint in SQL text (critical for performance/row count).
        # Prefer ORDER BY+LIMIT when max_rows is set to avoid fetching huge boxes from full Gaia DB.
        lim = None
        if max_rows is not None:
            try:
                lim_i = int(max_rows)
                if lim_i > 0:
                    lim = lim_i
            except (TypeError, ValueError):
                lim = None
        if lim is not None:
            # With proper indexes, ORDER BY g_mag gives a stable "brightest-first" subset.
            query = (
                f"SELECT {sel} FROM gaia_dr3 "
                f"WHERE ra >= ? AND ra <= ? AND dec >= ? AND dec <= ? AND g_mag <= {float(mag_limit)} "
                f"ORDER BY g_mag ASC LIMIT {int(lim)};"
            )
        else:
            query = (
                f"SELECT {sel} FROM gaia_dr3 "
                f"WHERE ra >= ? AND ra <= ? AND dec >= ? AND dec <= ? AND g_mag <= {float(mag_limit)};"
            )
        cur = conn.execute(query, (float(ra_min), float(ra_max), float(dec_min), float(dec_max)))
        rows = [dict(r) for r in cur.fetchall()]
        try:
            log_event(f"GAIA SQL: Found {len(rows)} stars (Mag <= {float(mag_limit)})")
        except Exception:  # noqa: BLE001
            pass
        return rows
    finally:
        conn.close()


def validate_gaia_db_schema(db_path: str | Path) -> tuple[bool, str]:
    """Validate local Gaia DB has table/columns required by VYVAR."""
    p = Path(db_path).expanduser().resolve()
    if not str(db_path).strip() or not p.is_file():
        return False, "missing_file"
    con = sqlite3.connect(str(p))
    try:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='gaia_dr3' LIMIT 1;"
        )
        if cur.fetchone() is None:
            return False, "missing_table_gaia_dr3"
        cur2 = con.execute("PRAGMA table_info('gaia_dr3');")
        cols = {str(r[1]).strip().lower() for r in cur2.fetchall()}
        required = {"source_id", "ra", "dec", "g_mag", "bp_rp"}
        missing = sorted([c for c in required if c not in cols])
        if missing:
            return False, f"missing_columns:{','.join(missing)}"
        return True, "ok"
    finally:
        con.close()


def validate_vsx_local_db_schema(db_path: str | Path) -> tuple[bool, str]:
    """Validate local VSX subset SQLite (``vsx_data`` from VizieR B/vsx/vsx import)."""
    p = Path(db_path).expanduser().resolve()
    if not str(db_path).strip() or not p.is_file():
        return False, "missing_file"
    con = sqlite3.connect(str(p))
    try:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vsx_data' LIMIT 1;"
        )
        if cur.fetchone() is None:
            return False, "missing_table_vsx_data"
        cur2 = con.execute("PRAGMA table_info('vsx_data');")
        cols = {str(r[1]).strip().lower() for r in cur2.fetchall()}
        required = {"oid", "ra_deg", "dec_deg"}
        missing = sorted([c for c in required if c not in cols])
        if missing:
            return False, f"missing_columns:{','.join(missing)}"
        return True, "ok"
    finally:
        con.close()


def _vsx_ra_intervals_deg(ra_min: float, ra_max: float) -> list[tuple[float, float]]:
    """Split an RA range (deg) into sub-intervals within [0, 360) when the box crosses the meridian."""
    rm, rM = float(ra_min), float(ra_max)
    if rm >= 0.0 and rM <= 360.0 and rm <= rM:
        return [(rm, rM)]
    out: list[tuple[float, float]] = []
    if rm < 0.0:
        out.append((360.0 + rm, 360.0))
        rm = 0.0
    if rM > 360.0:
        out.append((0.0, rM - 360.0))
        rM = 360.0
    if rm < rM:
        out.append((rm, rM))
    return out


def query_local_vsx(
    db_path: str | Path,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Query local VSX SQLite (``vsx_data``) for a rectangular RA/Dec window (deg, ICRS).

    Uses the same bounding box as ``query_local_gaia``; RA wrap at 0° is handled via split intervals.
    Rows are de-duplicated by ``oid`` when present, else by (ra_deg, dec_deg).
    """
    p = Path(db_path).expanduser().resolve()
    if not p.is_file():
        return []
    de0 = float(dec_min)
    de1 = float(dec_max)
    if de1 < de0:
        de0, de1 = de1, de0
    de0 = max(-90.0, min(90.0, de0))
    de1 = max(-90.0, min(90.0, de1))
    intervals = _vsx_ra_intervals_deg(float(ra_min), float(ra_max))
    lim: int | None = None
    if max_rows is not None:
        try:
            lim_i = int(max_rows)
            lim = lim_i if lim_i > 0 else None
        except (TypeError, ValueError):
            lim = None

    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vsx_ra ON vsx_data (ra_deg);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vsx_dec ON vsx_data (dec_deg);")
            conn.commit()
        except Exception:  # noqa: BLE001
            pass
        cur_cols = conn.execute("PRAGMA table_info('vsx_data');")
        cols = {str(r[1]).strip().lower() for r in cur_cols.fetchall()}
        if not cols or "ra_deg" not in cols or "dec_deg" not in cols:
            return []
        # VSX schema differs by source. Prefer period columns when present.
        want = [
            "oid",
            "name",
            "ra_deg",
            "dec_deg",
            "var_type",
            # Optional period columns (if present in vsx_data)
            "period",
            "varperiod",
            "var_period",
            "mag_max",
            "mag_min",
        ]
        sel_cols = [c for c in want if c in cols]
        sel = ", ".join(sel_cols)
        seen: set[Any] = set()
        rows_out: list[dict[str, Any]] = []
        for rlo, rhi in intervals:
            if lim is not None and len(rows_out) >= lim:
                break
            q = (
                f"SELECT {sel} FROM vsx_data "
                "WHERE ra_deg >= ? AND ra_deg <= ? AND dec_deg >= ? AND dec_deg <= ?;"
            )
            cur = conn.execute(q, (float(rlo), float(rhi), de0, de1))
            for r in cur.fetchall():
                d = dict(r)
                oid = d.get("oid")
                key: Any
                if oid is not None:
                    key = oid
                else:
                    key = (d.get("ra_deg"), d.get("dec_deg"))
                if key in seen:
                    continue
                seen.add(key)
                rows_out.append(d)
                if lim is not None and len(rows_out) >= lim:
                    break
        try:
            log_event(
                f"VSX SQL: {len(rows_out)} riadkov (obdĺžnik Dec=[{de0:.3f},{de1:.3f}], RA intervaly={len(intervals)})"
            )
        except Exception:  # noqa: BLE001
            pass
        return rows_out
    finally:
        conn.close()


class DraftTechnicalMetadataError(RuntimeError):
    """Focal length and/or effective pixel pitch missing after FITS + OBS_DRAFT SQL merge."""

    def __init__(self, draft_id: int) -> None:
        self.draft_id = int(draft_id)
        super().__init__(
            f"Kritická chyba: Chýbajú technické parametre pre Draft {self.draft_id}. "
            "Skontrolujte tabuľky EQUIPMENTS a TELESCOPES."
        )


def _db_fits_pixel_raw_to_micrometres(value: float) -> float:
    """Map raw FITS pixel-size keywords to micrometres (same semantics as ``pipeline``)."""
    if not math.isfinite(value) or value <= 0:
        return 0.0
    v = float(value)
    if v < 5e-5:
        return v * 1e6
    if v < 0.2:
        return v * 1000.0
    return v


def _db_to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _db_pick_header(header: fits.Header, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return default


def _db_xbinning_strict(header: fits.Header) -> int:
    """``XBINNING`` only; default 1 if absent (per draft/optics spec)."""
    if "XBINNING" not in header or header["XBINNING"] in (None, "", " ", "0", 0):
        return 1
    try:
        v = int(float(header["XBINNING"]))
    except (TypeError, ValueError):
        return 1
    return max(1, v)


def _db_ybinning_header(header: fits.Header, x_fallback: int) -> int:
    raw = _db_pick_header(header, "YBINNING", default=None)
    if raw in (None, "", " ", "0", 0):
        return max(1, int(x_fallback))
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        return max(1, int(x_fallback))
    return max(1, v)


def _db_focal_plausible_mm(mm: float) -> bool:
    return math.isfinite(mm) and 40.0 <= mm <= 120_000.0


def _db_header_focal_length_mm(header: fits.Header) -> float | None:
    """Focal length [mm] from FITS; ``FOCALLEN`` / ``FOCLEN`` often metres."""
    for key in ("FOCALLEN", "FOCLEN", "TELFOCA", "FOCAL_LEN", "FOCALL", "FOC_LEN"):
        if key not in header or header[key] in (None, "", " ", "0", 0):
            continue
        try:
            v = float(header[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        mm = v * 1000.0 if v < 25.0 else v
        if _db_focal_plausible_mm(mm):
            return float(mm)
    return None


def _db_header_pixel_native_um_mean(header: fits.Header) -> float | None:
    p1 = _db_fits_pixel_raw_to_micrometres(
        _db_to_float(_db_pick_header(header, "PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0))
    )
    p2 = _db_fits_pixel_raw_to_micrometres(
        _db_to_float(_db_pick_header(header, "PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0))
    )
    px = p1 if p1 > 0 else None
    py = p2 if p2 > 0 else None
    if px is not None and py is not None:
        return (float(px) + float(py)) / 2.0
    if px is not None:
        return float(px)
    if py is not None:
        return float(py)
    return None


def fits_header_cache_row_to_meta(row: sqlite3.Row) -> dict[str, Any]:
    """Rebuild ``extract_fits_metadata``-style dict from ``FITS_HEADER_CACHE`` row."""
    pu = row["PIXEL_UM"]
    tel, cam = row["TELESCOPE"], row["CAMERA"]
    pitch: float | None
    if pu is None:
        pitch = None
    else:
        try:
            pf = float(pu)
            pitch = pf if math.isfinite(pf) else None
        except (TypeError, ValueError):
            pitch = None
    bx = max(1, int(row["BINNING"] or 1))
    by = int(row["BINNING_Y"] if row["BINNING_Y"] is not None else row["BINNING"] or 1)
    by = max(1, by)
    phys_approx: float | None = None
    if pitch is not None and bx > 0:
        try:
            phys_approx = float(pitch) / float(bx)
            if not math.isfinite(phys_approx):
                phys_approx = None
        except (TypeError, ValueError):
            phys_approx = None
    return {
        "exposure": float(row["EXPTIME"] if row["EXPTIME"] is not None else 0.0),
        "filter": str(row["FILTER"] or "NoFilter"),
        "binning": bx,
        "binning_y": by,
        "naxis1": int(row["NAXIS1"] or 0),
        "naxis2": int(row["NAXIS2"] or 0),
        "pixel_size_um_physical": phys_approx,
        "pixel_size_um_header": pitch,
        "effective_pixel_um_plate_scale": pitch,
        "temp": float(row["CCD_TEMP"] if row["CCD_TEMP"] is not None else 0.0),
        "gain": int(row["GAIN"] or 0),
        "ra": float(row["RA_DEG"] if row["RA_DEG"] is not None else 0.0),
        "dec": float(row["DEC_DEG"] if row["DEC_DEG"] is not None else 0.0),
        "jd_start": float(row["JD_START"] if row["JD_START"] is not None else 0.0),
        "telescope": None if tel in (None, "") else str(tel),
        "camera": None if cam in (None, "") else str(cam),
    }


def _fits_header_cache_pack_row(
    fp: Path,
    sz: int,
    mt: float,
    meta: dict[str, Any],
    imagetyp: str,
    date_obs: str | None,
) -> tuple[Any, ...]:
    bx = int(meta.get("binning", 1) or 1)
    by = int(meta.get("binning_y", bx) or bx)
    bx = max(1, bx)
    by = max(1, by)
    # ``PIXEL_UM`` stores **effective** on-sky pitch [µm] (physical header pixel × binning).
    pu = meta.get("pixel_size_um_header")
    if pu is None:
        pphys = meta.get("pixel_size_um_physical")
        if pphys is not None:
            try:
                ppv = float(pphys)
                if math.isfinite(ppv) and ppv > 0:
                    pu = ppv * float(bx)
            except (TypeError, ValueError):
                pu = None
    pu_sql: float | None
    if pu is None:
        pu_sql = None
    else:
        try:
            pfv = float(pu)
            pu_sql = pfv if math.isfinite(pfv) else None
        except (TypeError, ValueError):
            pu_sql = None
    tel = meta.get("telescope")
    cam = meta.get("camera")
    return (
        str(fp.resolve()),
        int(sz),
        float(mt),
        float(meta.get("exposure", 0.0)),
        str(meta.get("filter", "NoFilter")),
        bx,
        by,
        int(meta.get("naxis1", 0) or 0),
        int(meta.get("naxis2", 0) or 0),
        pu_sql,
        float(meta.get("temp", 0.0)),
        int(meta.get("gain", 0) or 0),
        float(meta.get("ra", 0.0)),
        float(meta.get("dec", 0.0)),
        float(meta.get("jd_start", 0.0)),
        None if tel in (None, "") else str(tel),
        None if cam in (None, "") else str(cam),
        date_obs,
        str(imagetyp or ""),
    )


class VyvarDatabase:
    """Database manager for variable-star observation metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._enable_foreign_keys()
        self._create_tables()
        self.initialize_database()
        self._ensure_obs_files_indexes()
        self._ensure_settings_table()
        self._seed_default_settings()

    def _enable_foreign_keys(self) -> None:
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS EQUIPMENTS (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                CAMERANAME TEXT,
                ALIAS TEXT,
                SENSORTYPE TEXT,
                SENSORSIZE TEXT,
                PIXELSIZE REAL,
                ACTIVE TEXT DEFAULT 'YES'
            );

            CREATE TABLE IF NOT EXISTS TELESCOPE (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                TELESCOPENAME TEXT,
                ALIAS TEXT,
                DIAMETER REAL,
                FOCAL REAL,
                ACTIVE INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS LOCATION (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                PLACENAME TEXT,
                LATITUDE REAL,
                LONGITUDE REAL,
                ALTITUDE REAL
            );

            CREATE TABLE IF NOT EXISTS SCANNING (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                EXPTIME REAL,
                FILTERS TEXT,
                BINNING INTEGER,
                SENSORTEMP REAL,
                GAIN INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS OBSERVATION (
                ID TEXT PRIMARY KEY,
                ID_EQUIPMENTS INTEGER,
                ID_TELESCOPE INTEGER,
                ID_LOCATION INTEGER,
                ID_SCANNING INTEGER,
                CENTEROFFIELDRA REAL,
                CENTEROFFIELDDE REAL,
                OBSERVATIONSTARTJD REAL,
                FOREIGN KEY (ID_EQUIPMENTS) REFERENCES EQUIPMENTS (ID),
                FOREIGN KEY (ID_TELESCOPE) REFERENCES TELESCOPE (ID),
                FOREIGN KEY (ID_LOCATION) REFERENCES LOCATION (ID),
                FOREIGN KEY (ID_SCANNING) REFERENCES SCANNING (ID)
            );

            CREATE TABLE IF NOT EXISTS OBS_DRAFT (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                ID_EQUIPMENTS INTEGER,
                ID_TELESCOPE INTEGER,
                ID_LOCATION INTEGER,
                ID_SCANNING INTEGER,
                OBSERVATIONSTARTJD REAL,
                CENTEROFFIELDRA REAL,
                CENTEROFFIELDDE REAL,
                STATUS TEXT DEFAULT 'INGESTED',
                FINAL_OBSERVATION_ID TEXT,
                LIGHTS_PATH TEXT,
                CALIB_PATH TEXT,
                IMPORTED_AT TEXT,
                IMPORT_WARNINGS TEXT,
                IS_CALIBRATED INTEGER,
                ARCHIVE_PATH TEXT,
                FOREIGN KEY (ID_EQUIPMENTS) REFERENCES EQUIPMENTS (ID),
                FOREIGN KEY (ID_TELESCOPE) REFERENCES TELESCOPE (ID),
                FOREIGN KEY (ID_LOCATION) REFERENCES LOCATION (ID),
                FOREIGN KEY (ID_SCANNING) REFERENCES SCANNING (ID),
                FOREIGN KEY (FINAL_OBSERVATION_ID) REFERENCES OBSERVATION (ID)
            );

            CREATE TABLE IF NOT EXISTS OBS_FILES (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                OBSERVATION_ID TEXT,
                DRAFT_ID INTEGER,
                FILE_PATH TEXT,
                IMAGETYP TEXT,
                FILTER TEXT,
                FOREIGN KEY (OBSERVATION_ID) REFERENCES OBSERVATION (ID) ON DELETE CASCADE,
                FOREIGN KEY (DRAFT_ID) REFERENCES OBS_DRAFT (ID) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()
        self._ensure_active_columns()
        self._ensure_equipments_saturate_adu_column()
        self._ensure_equipments_cosmic_columns()
        self._ensure_equipments_focal_column()
        self._ensure_scanning_gain_column()
        self._ensure_obs_files_draft_column()
        self._ensure_obs_files_qc_columns()
        self._ensure_obs_files_quality_inspection_columns()
        self._ensure_obs_files_observation_group_key()
        self._ensure_obs_files_scanning_and_calibration_columns()
        self._ensure_photometry_light_curve_table()
        self._ensure_observation_import_columns()
        self._ensure_calibration_library_table()
        self._ensure_fits_header_cache_table()
        self._ensure_qc_processing_tables()
        self._ensure_final_data_view()
        self._ensure_master_sources_table()
        self._ensure_obs_draft_masterstar_path_column()
        self._ensure_obs_draft_status_panel_columns()

    def _ensure_final_data_view(self) -> None:
        """View of rows that reference equipment/telescope (drafts, observations, QC runs).

        Used for hard-delete integrity (hashtag / final pipeline safety). Not a physical table.
        """
        cur = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='view' AND name='FINAL_DATA' LIMIT 1;"
        )
        if cur.fetchone() is not None:
            return
        self.conn.execute(
            """
            CREATE VIEW FINAL_DATA AS
            SELECT d.ID_EQUIPMENTS AS equipment_id, d.ID_TELESCOPE AS telescope_id
            FROM OBS_DRAFT d
            UNION ALL
            SELECT o.ID_EQUIPMENTS, o.ID_TELESCOPE
            FROM OBSERVATION o
            UNION ALL
            SELECT d2.ID_EQUIPMENTS, d2.ID_TELESCOPE
            FROM OBS_QC_PROCESSING_RUN r
            INNER JOIN OBS_DRAFT d2 ON d2.ID = r.DRAFT_ID;
            """
        )
        self.conn.commit()

    def _ensure_master_sources_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS MASTER_SOURCES (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                DRAFT_ID INTEGER,
                SOURCE_ID_GAIA TEXT,
                X_MASTER REAL,
                Y_MASTER REAL,
                RA REAL,
                DE REAL,
                G_MAG REAL,
                BP_RP REAL,
                G_FLUX_ERROR_REL REAL,
                NON_SINGLE_STAR INTEGER DEFAULT 0,
                PHOT_VARIABLE_FLAG TEXT,
                FILTER_NAME TEXT,
                PHOT_CATEGORY TEXT,
                RECOMMENDED_APERTURE REAL,
                IS_VAR INTEGER DEFAULT 0,
                IS_SATURATED INTEGER DEFAULT 0,
                IS_SAFE_COMP INTEGER DEFAULT 1,
                EXCLUSION_REASON TEXT,
                STRESS_RMS REAL,
                SAFE_OVERRIDE INTEGER DEFAULT 0,
                CREATED_AT TEXT NOT NULL,
                FOREIGN KEY (DRAFT_ID) REFERENCES OBS_DRAFT (ID) ON DELETE SET NULL
            );
            """
        )
        # Migrations for older DBs: add missing columns.
        cur = self.conn.execute("PRAGMA table_info('MASTER_SOURCES');")
        cols = {str(r[1]).upper() for r in cur.fetchall()}
        add_cols: list[tuple[str, str]] = []
        if "FILTER_NAME" not in cols:
            add_cols.append(("FILTER_NAME", "TEXT"))
        if "PHOT_CATEGORY" not in cols:
            add_cols.append(("PHOT_CATEGORY", "TEXT"))
        if "G_FLUX_ERROR_REL" not in cols:
            add_cols.append(("G_FLUX_ERROR_REL", "REAL"))
        if "NON_SINGLE_STAR" not in cols:
            add_cols.append(("NON_SINGLE_STAR", "INTEGER DEFAULT 0"))
        if "PHOT_VARIABLE_FLAG" not in cols:
            add_cols.append(("PHOT_VARIABLE_FLAG", "TEXT"))
        if "RECOMMENDED_APERTURE" not in cols:
            add_cols.append(("RECOMMENDED_APERTURE", "REAL"))
        if "IS_SAFE_COMP" not in cols:
            add_cols.append(("IS_SAFE_COMP", "INTEGER DEFAULT 1"))
        if "EXCLUSION_REASON" not in cols:
            add_cols.append(("EXCLUSION_REASON", "TEXT"))
        if "STRESS_RMS" not in cols:
            add_cols.append(("STRESS_RMS", "REAL"))
        if "SAFE_OVERRIDE" not in cols:
            add_cols.append(("SAFE_OVERRIDE", "INTEGER DEFAULT 0"))
        if "LIKELY_NONLINEAR" not in cols:
            add_cols.append(("LIKELY_NONLINEAR", "INTEGER DEFAULT 0"))
        if "ON_BAD_COLUMN" not in cols:
            add_cols.append(("ON_BAD_COLUMN", "INTEGER DEFAULT 0"))
        for name, sql_type in add_cols:
            self.conn.execute(f"ALTER TABLE MASTER_SOURCES ADD COLUMN {name} {sql_type};")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_MASTER_SOURCES_DRAFT ON MASTER_SOURCES (DRAFT_ID);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_MASTER_SOURCES_GAIA ON MASTER_SOURCES (SOURCE_ID_GAIA);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_MASTER_SOURCES_PHOTCAT ON MASTER_SOURCES (DRAFT_ID, PHOT_CATEGORY);"
        )
        self.conn.commit()

    def replace_master_sources_for_draft(self, draft_id: int, rows: list[dict[str, Any]]) -> int:
        """Replace MASTER_SOURCES rows for a draft (delete+insert)."""
        did = int(draft_id)
        self.conn.execute("DELETE FROM MASTER_SOURCES WHERE DRAFT_ID = ?;", (did,))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n = 0
        for r in rows:
            self.conn.execute(
                """
                INSERT INTO MASTER_SOURCES (
                    DRAFT_ID,
                    SOURCE_ID_GAIA,
                    X_MASTER,
                    Y_MASTER,
                    RA,
                    DE,
                    G_MAG,
                    BP_RP,
                    G_FLUX_ERROR_REL,
                    NON_SINGLE_STAR,
                    PHOT_VARIABLE_FLAG,
                    FILTER_NAME,
                    PHOT_CATEGORY,
                    RECOMMENDED_APERTURE,
                    IS_VAR,
                    IS_SATURATED,
                    IS_SAFE_COMP,
                    EXCLUSION_REASON,
                    STRESS_RMS,
                    SAFE_OVERRIDE,
                    LIKELY_NONLINEAR,
                    ON_BAD_COLUMN,
                    CREATED_AT
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    did,
                    str(r.get("source_id_gaia") or ""),
                    float(r.get("x_master")) if r.get("x_master") is not None else None,
                    float(r.get("y_master")) if r.get("y_master") is not None else None,
                    float(r.get("ra")) if r.get("ra") is not None else None,
                    float(r.get("dec")) if r.get("dec") is not None else None,
                    float(r.get("g_mag")) if r.get("g_mag") is not None else None,
                    float(r.get("bp_rp")) if r.get("bp_rp") is not None else None,
                    float(r.get("g_flux_error_rel")) if r.get("g_flux_error_rel") is not None else None,
                    1 if int(r.get("non_single_star") or 0) else 0,
                    str(r.get("phot_variable_flag") or "") or None,
                    str(r.get("filter_name") or ""),
                    str(r.get("phot_category") or ""),
                    float(r.get("recommended_aperture")) if r.get("recommended_aperture") is not None else None,
                    1 if int(r.get("is_var") or 0) else 0,
                    1 if int(r.get("is_saturated") or 0) else 0,
                    1 if int(r.get("is_safe_comp") if r.get("is_safe_comp") is not None else 1) else 0,
                    str(r.get("exclusion_reason") or "") or None,
                    float(r.get("stress_rms")) if r.get("stress_rms") is not None else None,
                    1 if int(r.get("safe_override") or 0) else 0,
                    1 if int(r.get("likely_nonlinear") or 0) else 0,
                    1 if int(r.get("on_bad_column") or 0) else 0,
                    now,
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def fetch_master_sources_for_draft(self, draft_id: int) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT *
            FROM MASTER_SOURCES
            WHERE DRAFT_ID = ?
            ORDER BY
              COALESCE(IS_SAFE_COMP, 0) DESC,
              COALESCE(STRESS_RMS, 1e9) ASC,
              COALESCE(G_MAG, 99) ASC,
              ID ASC;
            """,
            (int(draft_id),),
        )
        return [dict(r) for r in cur.fetchall()]

    def update_master_source_safety(
        self,
        source_row_id: int,
        *,
        is_safe_comp: bool,
        exclusion_reason: str | None = None,
        safe_override: bool = True,
    ) -> None:
        self.conn.execute(
            """
            UPDATE MASTER_SOURCES
            SET IS_SAFE_COMP = ?,
                EXCLUSION_REASON = ?,
                SAFE_OVERRIDE = ?
            WHERE ID = ?;
            """,
            (
                1 if bool(is_safe_comp) else 0,
                (str(exclusion_reason).strip() if exclusion_reason is not None else None),
                1 if bool(safe_override) else 0,
                int(source_row_id),
            ),
        )
        self.conn.commit()

    @staticmethod
    def sql_expr_active_is_true(column_ref: str) -> str:
        """SQL predicate: row is *active* (soft-delete off).

        Accepts ``NULL``, numeric ``1``, or text ``YES``/``1``/``TRUE``/``Y`` as active;
        numeric ``0`` or text ``NO``/``0``/``FALSE`` as inactive. Aligns with ``ACTIVE = 1`` style checks.
        """
        c = column_ref.strip()
        return f"""(
            ({c}) IS NULL
            OR (typeof({c}) IN ('integer', 'real') AND CAST({c} AS INTEGER) != 0)
            OR (typeof({c}) = 'text' AND UPPER(TRIM(CAST({c} AS TEXT))) IN ('YES', 'Y', 'TRUE', '1'))
        )"""

    def count_final_data_for_equipment_id(self, equipment_id: int) -> int:
        self._ensure_final_data_view()
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM FINAL_DATA WHERE equipment_id = ?;",
            (int(equipment_id),),
        )
        row = cur.fetchone()
        return int(row[0]) if row is not None else 0

    def count_final_data_for_telescope_id(self, telescope_id: int) -> int:
        self._ensure_final_data_view()
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM FINAL_DATA WHERE telescope_id = ?;",
            (int(telescope_id),),
        )
        row = cur.fetchone()
        return int(row[0]) if row is not None else 0

    def count_references_to_location_id(self, location_id: int) -> int:
        n = 0
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM OBS_DRAFT WHERE ID_LOCATION = ?;",
            (int(location_id),),
        )
        n += int(cur.fetchone()[0])
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM OBSERVATION WHERE ID_LOCATION = ?;",
            (int(location_id),),
        )
        n += int(cur.fetchone()[0])
        return n

    def count_references_to_scanning_id(self, scanning_id: int) -> int:
        n = 0
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM OBS_DRAFT WHERE ID_SCANNING = ?;",
            (int(scanning_id),),
        )
        n += int(cur.fetchone()[0])
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM OBSERVATION WHERE ID_SCANNING = ?;",
            (int(scanning_id),),
        )
        n += int(cur.fetchone()[0])
        return n

    @staticmethod
    def normalize_active_db_value(raw: Any) -> int:
        """Normalize UI / DB values to **1** = active, **0** = inactive (soft-delete). Legacy ``YES``/``NO`` supported."""
        if raw is None:
            return 1
        try:
            if isinstance(raw, float) and math.isnan(raw):
                return 1
        except TypeError:
            pass
        if raw is True:
            return 1
        if raw is False:
            return 0
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            try:
                return 0 if int(raw) == 0 else 1
            except (TypeError, ValueError):
                return 1
        s = str(raw).strip().upper()
        if s in ("NO", "N", "FALSE", "0", "0.0"):
            return 0
        if s in ("YES", "Y", "TRUE", "1", "1.0", ""):
            return 1
        return 1

    def _coerce_sql_param(self, table: str, col: str, raw: Any) -> Any:
        if col == "ACTIVE":
            n = int(self.normalize_active_db_value(raw))
            if table == "TELESCOPE":
                return 0 if n == 0 else 1
            return n
        if raw is None:
            return None
        try:
            if isinstance(raw, float) and math.isnan(raw):
                return None
        except TypeError:
            pass
        try:
            import pandas as pd_na

            if pd_na.isna(raw):
                return None
        except Exception:
            pass
        return raw

    def apply_main_table_editor_save(
        self,
        table: str,
        pk_col: str,
        original_df: Any,
        edited_df: Any,
        *,
        editable_cols: list[str],
    ) -> dict[str, int]:
        """Apply INSERT/UPDATE/DELETE from a Streamlit ``data_editor`` diff.

        Only ``EQUIPMENTS``, ``TELESCOPE``, ``SCANNING``, ``LOCATION`` are allowed.

        **EQUIPMENTS / TELESCOPE:** a row removed from the editor is **not** ``DELETE``-d; ``ACTIVE`` is set
        to **0** (soft-delete). Physical ``DELETE`` for those tables is never performed from this API.
        **LOCATION / SCANNING:** physical delete with reference checks (``FINAL_DATA`` / FK usage).
        """
        import pandas as pd_local

        allowed = {"EQUIPMENTS", "TELESCOPE", "SCANNING", "LOCATION"}
        if table not in allowed:
            raise ValueError(f"Table not allowed for universal editor: {table}")
        if pk_col != "ID":
            raise ValueError("Expected primary key column ID.")

        def _parse_pk(val: Any) -> int | None:
            if val is None:
                return None
            try:
                if isinstance(val, float) and math.isnan(val):
                    return None
            except TypeError:
                pass
            if pd_local.isna(val):
                return None
            s = str(val).strip()
            if not s:
                return None
            try:
                return int(float(s))
            except (TypeError, ValueError):
                return None

        orig = original_df.copy()
        edit = edited_df.copy()
        if pk_col not in orig.columns or pk_col not in edit.columns:
            raise ValueError(f"Missing primary key column {pk_col!r}.")

        orig_ids: set[int] = set()
        for v in orig[pk_col].tolist():
            pid = _parse_pk(v)
            if pid is not None:
                orig_ids.add(pid)

        edited_by_id: dict[int, Any] = {}
        new_rows: list[Any] = []
        for _, row in edit.iterrows():
            pid = _parse_pk(row.get(pk_col))
            if pid is None:
                new_rows.append(row)
            else:
                edited_by_id[pid] = row

        deleted_ids = orig_ids - set(edited_by_id.keys())
        inserted = 0
        updated = 0
        deleted = 0
        soft_deactivated = 0

        pragma_cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table});").fetchall()]
        insert_colnames = [c for c in pragma_cols if c != pk_col]

        try:
            self.conn.execute("BEGIN;")
            for did in sorted(deleted_ids):
                if table in ("EQUIPMENTS", "TELESCOPE"):
                    if "ACTIVE" not in pragma_cols:
                        raise ValueError(f"Tabuľka {table} nemá stĺpec ACTIVE — soft-delete nie je možný.")
                    self.conn.execute(
                        f"UPDATE {table} SET ACTIVE = 0 WHERE {pk_col} = ?;",
                        (did,),
                    )
                    soft_deactivated += 1
                    continue
                if table == "LOCATION":
                    n = self.count_references_to_location_id(did)
                    if n > 0:
                        raise ValueError(
                            f"Lokalitu ID={did} nie je možné zmazať: {n} odkazov v OBS_DRAFT/OBSERVATION."
                        )
                elif table == "SCANNING":
                    n = self.count_references_to_scanning_id(did)
                    if n > 0:
                        raise ValueError(
                            f"Scanning ID={did} nie je možné zmazať: {n} odkazov v OBS_DRAFT/OBSERVATION."
                        )
                self.conn.execute(f"DELETE FROM {table} WHERE {pk_col} = ?;", (did,))
                deleted += 1

            for pid, row in edited_by_id.items():
                if pid not in orig_ids:
                    continue
                hits = orig.loc[orig[pk_col].apply(_parse_pk) == pid]
                if hits.empty:
                    continue
                orig_row = hits.iloc[0]
                changes: dict[str, Any] = {}
                for col in editable_cols:
                    if col == pk_col or col not in pragma_cols:
                        continue
                    if col not in row.index or col not in orig_row.index:
                        continue
                    nv = self._coerce_sql_param(table, col, row[col])
                    ov = self._coerce_sql_param(table, col, orig_row[col])
                    if nv != ov:
                        changes[col] = nv
                if not changes:
                    continue
                set_sql = ", ".join(f"{k} = ?" for k in changes.keys())
                params = list(changes.values()) + [pid]
                self.conn.execute(f"UPDATE {table} SET {set_sql} WHERE {pk_col} = ?;", params)
                updated += 1

            for row in new_rows:
                vals: list[Any] = []
                for c in insert_colnames:
                    raw = row[c] if c in row.index else None
                    vals.append(self._coerce_sql_param(table, c, raw))
                placeholders = ", ".join("?" * len(insert_colnames))
                cols_sql = ", ".join(insert_colnames)
                self.conn.execute(
                    f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders});",
                    vals,
                )
                inserted += 1

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return {
            "inserted": inserted,
            "updated": updated,
            "deleted": deleted,
            "soft_deactivated": soft_deactivated,
        }

    def _ensure_qc_processing_tables(self) -> None:
        """Append-only QC Apply snapshots: hashtag + accepted ``OBS_FILES`` rows (IS_REJECTED=0)."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS OBS_QC_PROCESSING_RUN (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                PROCESSING_HASH TEXT NOT NULL UNIQUE,
                DRAFT_ID INTEGER NOT NULL,
                CREATED_AT TEXT NOT NULL,
                FOREIGN KEY (DRAFT_ID) REFERENCES OBS_DRAFT (ID) ON DELETE CASCADE
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS OBS_QC_PROCESSING_FILE (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                RUN_ID INTEGER NOT NULL,
                SOURCE_OBS_FILE_ID INTEGER NOT NULL,
                FILE_PATH TEXT,
                FILTER TEXT,
                EXPTIME REAL,
                INSPECTION_JD REAL,
                FWHM REAL,
                DRIFT REAL,
                FOREIGN KEY (RUN_ID) REFERENCES OBS_QC_PROCESSING_RUN (ID) ON DELETE CASCADE
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_QC_PROC_RUN_DRAFT ON OBS_QC_PROCESSING_RUN (DRAFT_ID);"
        )
        self.conn.commit()

    def fetch_obs_draft_by_id(self, draft_id: int) -> dict[str, Any] | None:
        cur = self.conn.execute("SELECT * FROM OBS_DRAFT WHERE ID = ?;", (int(draft_id),))
        row = cur.fetchone()
        return dict(row) if row is not None else None

    def update_obs_draft_center(self, draft_id: int, center_ra_deg: float, center_de_deg: float) -> None:
        """Persist draft field center (ICRS degrees) in ``OBS_DRAFT``."""
        cur = self.conn.execute(
            """
            UPDATE OBS_DRAFT
               SET CENTEROFFIELDRA = ?,
                   CENTEROFFIELDDE = ?
             WHERE ID = ?;
            """,
            (float(center_ra_deg), float(center_de_deg), int(draft_id)),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Draft '{draft_id}' not found for center update.")
        self.conn.commit()

    def update_obs_draft_status_panel_values(
        self,
        draft_id: int,
        *,
        center_ra_deg: float | None = None,
        center_de_deg: float | None = None,
        focal_mm: float | None = None,
        pixel_um: float | None = None,
    ) -> None:
        """Persist status-panel values for a draft (center + optional focal/pixel)."""
        sets: list[str] = []
        params: list[Any] = []
        if center_ra_deg is not None:
            sets.append("CENTEROFFIELDRA = ?")
            params.append(float(center_ra_deg))
        if center_de_deg is not None:
            sets.append("CENTEROFFIELDDE = ?")
            params.append(float(center_de_deg))
        if focal_mm is not None:
            sets.append("FOCAL_MM = ?")
            params.append(float(focal_mm))
        if pixel_um is not None:
            sets.append("PIXEL_UM = ?")
            params.append(float(pixel_um))
        if not sets:
            return
        params.append(int(draft_id))
        cur = self.conn.execute(
            f"UPDATE OBS_DRAFT SET {', '.join(sets)} WHERE ID = ?;",
            tuple(params),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Draft '{draft_id}' not found for status-panel update.")
        self.conn.commit()

    def set_obs_draft_masterstar_path(self, draft_id: int, masterstar_path: str | None) -> None:
        """Persist absolute MASTERSTAR FITS path for a draft."""
        _p = (str(masterstar_path).strip() if masterstar_path is not None else "") or None
        cur = self.conn.execute(
            """
            UPDATE OBS_DRAFT
               SET MASTERSTAR_PATH = ?,
                   MASTERSTAR_FITS_PATH = ?
             WHERE ID = ?;
            """,
            (_p, _p, int(draft_id)),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Draft '{draft_id}' not found for MASTERSTAR path update.")
        self.conn.commit()

    def get_obs_draft_masterstar_path(self, draft_id: int) -> str | None:
        """Return persisted MASTERSTAR path for a draft (if available)."""
        cur = self.conn.execute(
            "SELECT MASTERSTAR_FITS_PATH, MASTERSTAR_PATH FROM OBS_DRAFT WHERE ID = ?;",
            (int(draft_id),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        try:
            v = row["MASTERSTAR_FITS_PATH"] or row["MASTERSTAR_PATH"]
        except Exception:  # noqa: BLE001
            v = row[0] if len(row) > 0 else None
        s = str(v).strip() if v is not None else ""
        return s or None

    def qc_processing_run_exists(self, processing_hash: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM OBS_QC_PROCESSING_RUN WHERE PROCESSING_HASH = ? LIMIT 1;",
            (str(processing_hash),),
        )
        return cur.fetchone() is not None

    def delete_qc_processing_run_by_hash(self, processing_hash: str) -> None:
        self.conn.execute(
            "DELETE FROM OBS_QC_PROCESSING_RUN WHERE PROCESSING_HASH = ?;",
            (str(processing_hash),),
        )
        self.conn.commit()

    def record_qc_processing_apply(
        self,
        draft_id: int,
        processing_hash: str,
        *,
        overwrite: bool = False,
    ) -> int:
        """Idempotent upsert run + copy light rows with ``IS_REJECTED`` 0."""
        h = str(processing_hash).strip()
        if not h:
            raise ValueError("processing_hash is empty")
        _ = overwrite  # backward compatible arg; run is always upserted
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.conn.execute(
            """
            INSERT INTO OBS_QC_PROCESSING_RUN (PROCESSING_HASH, DRAFT_ID, CREATED_AT)
            VALUES (?, ?, ?)
            ON CONFLICT(PROCESSING_HASH) DO UPDATE SET
                DRAFT_ID = excluded.DRAFT_ID,
                CREATED_AT = excluded.CREATED_AT;
            """,
            (h, int(draft_id), now),
        )
        run_row = self.conn.execute(
            "SELECT ID FROM OBS_QC_PROCESSING_RUN WHERE PROCESSING_HASH = ?;",
            (h,),
        ).fetchone()
        if run_row is None:
            raise ValueError(f"Cannot resolve QC processing run for hash: {h}")
        run_id = int(run_row["ID"])
        self.conn.execute("DELETE FROM OBS_QC_PROCESSING_FILE WHERE RUN_ID = ?;", (run_id,))
        cur2 = self.conn.execute(
            """
            SELECT ID, FILE_PATH, FILTER, EXPTIME, INSPECTION_JD, FWHM, DRIFT
            FROM OBS_FILES
            WHERE DRAFT_ID = ?
              AND LOWER(COALESCE(IMAGETYP, '')) = 'light'
              AND COALESCE(IS_REJECTED, 0) = 0;
            """,
            (int(draft_id),),
        )
        for frow in cur2.fetchall():
            ex_v: float | None = None
            if frow["EXPTIME"] is not None:
                try:
                    ex_f = float(frow["EXPTIME"])
                    ex_v = ex_f if math.isfinite(ex_f) else None
                except (TypeError, ValueError):
                    ex_v = None
            jd_v: float | None = None
            if frow["INSPECTION_JD"] is not None:
                try:
                    jd_f = float(frow["INSPECTION_JD"])
                    jd_v = jd_f if math.isfinite(jd_f) else None
                except (TypeError, ValueError):
                    jd_v = None
            fwhm_v: float | None = None
            if frow["FWHM"] is not None:
                try:
                    fw_f = float(frow["FWHM"])
                    fwhm_v = fw_f if math.isfinite(fw_f) else None
                except (TypeError, ValueError):
                    fwhm_v = None
            drift_v: float | None = None
            if frow["DRIFT"] is not None:
                try:
                    dr_f = float(frow["DRIFT"])
                    drift_v = dr_f if math.isfinite(dr_f) else None
                except (TypeError, ValueError):
                    drift_v = None
            self.conn.execute(
                """
                INSERT INTO OBS_QC_PROCESSING_FILE (
                    RUN_ID, SOURCE_OBS_FILE_ID, FILE_PATH, FILTER, EXPTIME, INSPECTION_JD, FWHM, DRIFT
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run_id,
                    int(frow["ID"]),
                    str(frow["FILE_PATH"] or ""),
                    str(frow["FILTER"]) if frow["FILTER"] is not None else None,
                    ex_v,
                    jd_v,
                    fwhm_v,
                    drift_v,
                ),
            )
        self.conn.commit()
        return run_id

    def update_obs_draft_status(self, draft_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE OBS_DRAFT SET STATUS = ? WHERE ID = ?;",
            (str(status), int(draft_id)),
        )
        self.conn.commit()

    def count_obs_files(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM OBS_FILES;")
        row = cur.fetchone()
        return int(row[0]) if row is not None else 0

    def maintenance_delete_obs_files_for_processed_drafts(self) -> int:
        """Delete ``OBS_FILES`` rows whose ``DRAFT_ID`` refers to ``OBS_DRAFT`` with ``STATUS = 'PROCESSED'``.

        Touches **only** ``OBS_FILES`` (not ``OBS_DRAFT``, not hashtag / equipment tables).
        """
        cur = self.conn.execute(
            """
            DELETE FROM OBS_FILES
            WHERE DRAFT_ID IN (
                SELECT ID FROM OBS_DRAFT
                WHERE UPPER(TRIM(COALESCE(STATUS, ''))) = 'PROCESSED'
            );
            """
        )
        n = int(cur.rowcount) if cur.rowcount is not None and cur.rowcount >= 0 else 0
        self.conn.commit()
        return n

    def maintenance_nuke_obs_files_and_drafts_preserve_qc_snapshots(self) -> tuple[int, int]:
        """Remove all rows from ``OBS_FILES`` and ``OBS_DRAFT`` (staging / draft workspace).

        Temporarily disables foreign-key enforcement only for the ``OBS_DRAFT`` delete so that
        ``OBS_QC_PROCESSING_RUN`` / ``OBS_QC_PROCESSING_FILE`` are **not** CASCADE-deleted.
        Leaves orphan ``DRAFT_ID`` values in QC runs (danger-zone tradeoff).

        Never executes SQL against ``EQUIPMENTS``, ``TELESCOPE``, ``OBSERVATION``, or ``OBS_QC_PROCESSING_*``.
        """
        cur_f = self.conn.execute("DELETE FROM OBS_FILES;")
        n_files = int(cur_f.rowcount) if cur_f.rowcount is not None and cur_f.rowcount >= 0 else 0
        self.conn.execute("PRAGMA foreign_keys = OFF;")
        try:
            cur_d = self.conn.execute("DELETE FROM OBS_DRAFT;")
            n_drafts = int(cur_d.rowcount) if cur_d.rowcount is not None and cur_d.rowcount >= 0 else 0
        finally:
            self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.commit()
        return (n_files, n_drafts)

    def _ensure_fits_header_cache_table(self) -> None:
        """Primary-header cache for fast smart_scan (path + size + mtime invalidation)."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS FITS_HEADER_CACHE (
                FILE_PATH TEXT PRIMARY KEY,
                FILE_SIZE INTEGER NOT NULL,
                MTIME REAL NOT NULL,
                EXPTIME REAL NOT NULL,
                FILTER TEXT NOT NULL,
                BINNING INTEGER NOT NULL,
                BINNING_Y INTEGER NOT NULL,
                NAXIS1 INTEGER NOT NULL,
                NAXIS2 INTEGER NOT NULL,
                PIXEL_UM REAL,
                CCD_TEMP REAL NOT NULL,
                GAIN INTEGER NOT NULL,
                RA_DEG REAL NOT NULL,
                DEC_DEG REAL NOT NULL,
                JD_START REAL NOT NULL,
                TELESCOPE TEXT,
                CAMERA TEXT,
                DATE_OBS TEXT,
                IMAGETYP TEXT NOT NULL DEFAULT ''
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_FITS_HDR_CACHE_STALE ON FITS_HEADER_CACHE (FILE_PATH, FILE_SIZE, MTIME);"
        )
        self.conn.commit()

    @staticmethod
    def fits_header_cache_file_key(path: str | Path) -> str:
        return str(Path(path).resolve())

    def fits_header_cache_get_if_fresh(
        self,
        path: str | Path,
        *,
        file_size: int,
        mtime: float,
    ) -> sqlite3.Row | None:
        """Return cache row if it exists and matches current ``file_size`` and ``mtime``."""
        key = self.fits_header_cache_file_key(path)
        row = self.conn.execute(
            "SELECT * FROM FITS_HEADER_CACHE WHERE FILE_PATH = ?;",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if int(row["FILE_SIZE"]) != int(file_size):
            return None
        # Filesystems may round mtime; allow tiny float noise (and ~2s coarse stamps).
        if abs(float(row["MTIME"]) - float(mtime)) > 2.0:
            return None
        return row

    def fits_header_cache_try_meta(self, path: str | Path) -> dict[str, Any] | None:
        """If file exists and cache row matches size+mtime, return metadata dict (``extract_fits_metadata`` shape)."""
        p = Path(path)
        try:
            st = p.stat()
        except OSError:
            return None
        row = self.fits_header_cache_get_if_fresh(p, file_size=int(st.st_size), mtime=float(st.st_mtime))
        if row is None:
            return None
        return fits_header_cache_row_to_meta(row)

    def fits_header_cache_try_filter(self, path: str | Path) -> str | None:
        """Return normalized filter string from cache if fresh; else ``None``."""
        p = Path(path)
        try:
            st = p.stat()
        except OSError:
            return None
        row = self.fits_header_cache_get_if_fresh(p, file_size=int(st.st_size), mtime=float(st.st_mtime))
        if row is None:
            return None
        flt = str(row["FILTER"] or "").strip()
        if not flt or flt.lower() in {"unknown", "none", "nan"}:
            return "NoFilter"
        return flt

    def fits_header_cache_upsert_batch(
        self,
        items: list[tuple[Path, int, float, dict[str, Any], str, str | None]],
    ) -> None:
        """Insert or replace many rows in one transaction. Each item:
        ``(path, file_size, mtime, meta_dict, imagetyp_raw, date_obs_str_or_none)``.
        """
        if not items:
            return
        sql = """
            INSERT INTO FITS_HEADER_CACHE (
                FILE_PATH, FILE_SIZE, MTIME, EXPTIME, FILTER, BINNING, BINNING_Y,
                NAXIS1, NAXIS2, PIXEL_UM, CCD_TEMP, GAIN, RA_DEG, DEC_DEG, JD_START,
                TELESCOPE, CAMERA, DATE_OBS, IMAGETYP
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(FILE_PATH) DO UPDATE SET
                FILE_SIZE = excluded.FILE_SIZE,
                MTIME = excluded.MTIME,
                EXPTIME = excluded.EXPTIME,
                FILTER = excluded.FILTER,
                BINNING = excluded.BINNING,
                BINNING_Y = excluded.BINNING_Y,
                NAXIS1 = excluded.NAXIS1,
                NAXIS2 = excluded.NAXIS2,
                PIXEL_UM = excluded.PIXEL_UM,
                CCD_TEMP = excluded.CCD_TEMP,
                GAIN = excluded.GAIN,
                RA_DEG = excluded.RA_DEG,
                DEC_DEG = excluded.DEC_DEG,
                JD_START = excluded.JD_START,
                TELESCOPE = excluded.TELESCOPE,
                CAMERA = excluded.CAMERA,
                DATE_OBS = excluded.DATE_OBS,
                IMAGETYP = excluded.IMAGETYP;
        """
        self.conn.execute("BEGIN IMMEDIATE;")
        try:
            for fp, sz, mt, meta, imagetyp, date_obs in items:
                self.conn.execute(sql, _fits_header_cache_pack_row(fp, sz, mt, meta, imagetyp, date_obs))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def fits_header_cache_upsert_one(
        self,
        path: Path,
        *,
        file_size: int,
        mtime: float,
        meta: dict[str, Any],
        imagetyp: str,
        date_obs: str | None,
    ) -> None:
        self.fits_header_cache_upsert_batch([(path, file_size, mtime, meta, imagetyp, date_obs)])

    def _ensure_calibration_library_table(self) -> None:
        """Master Dark/Flat registry for fast matching on import (path + XBINNING, EXPTIME, TEMP, FILTER)."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS CALIBRATION_LIBRARY (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                KIND TEXT NOT NULL,
                FILE_PATH TEXT NOT NULL UNIQUE,
                XBINNING INTEGER NOT NULL,
                EXPTIME REAL NOT NULL,
                CCD_TEMP REAL,
                FILTER_NAME TEXT NOT NULL DEFAULT '',
                GAIN INTEGER NOT NULL DEFAULT 0,
                NCOMBINE INTEGER,
                REGISTERED_AT TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_CAL_LIB_LOOKUP ON CALIBRATION_LIBRARY "
            "(KIND, XBINNING, EXPTIME, FILTER_NAME, GAIN);"
        )
        self.conn.commit()
        self._migrate_calibration_library_scope_columns()

    def _migrate_calibration_library_scope_columns(self) -> None:
        """Add optional equipment/telescope scope (set = kamera + ďalekohľad). Legacy rows: NULL,NULL = všeobecné."""
        cur = self.conn.execute("PRAGMA table_info('CALIBRATION_LIBRARY');")
        cols = {str(r[1]).upper() for r in cur.fetchall()}
        if "ID_EQUIPMENTS" not in cols:
            self.conn.execute("ALTER TABLE CALIBRATION_LIBRARY ADD COLUMN ID_EQUIPMENTS INTEGER;")
        if "ID_TELESCOPE" not in cols:
            self.conn.execute("ALTER TABLE CALIBRATION_LIBRARY ADD COLUMN ID_TELESCOPE INTEGER;")
        self.conn.commit()

    def register_calibration_library_entry(
        self,
        *,
        kind: str,
        file_path: str | Path,
        xbinning: int,
        exptime: float,
        ccd_temp: float | None,
        filter_name: str = "",
        gain: int = 0,
        ncombine: int | None = None,
        id_equipments: int | None = None,
        id_telescope: int | None = None,
    ) -> None:
        """Insert or update one master calibration row (keyed by absolute ``FILE_PATH``)."""
        k = str(kind or "").strip().lower()
        if k not in ("dark", "flat"):
            raise ValueError("kind must be 'dark' or 'flat'")
        fp = str(Path(file_path).resolve())
        flt = "" if k == "dark" else str(filter_name or "").strip()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _eq = int(id_equipments) if id_equipments is not None else None
        _tel = int(id_telescope) if id_telescope is not None else None
        self.conn.execute(
            """
            INSERT INTO CALIBRATION_LIBRARY (
                KIND, FILE_PATH, XBINNING, EXPTIME, CCD_TEMP, FILTER_NAME, GAIN, NCOMBINE,
                REGISTERED_AT, ID_EQUIPMENTS, ID_TELESCOPE
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(FILE_PATH) DO UPDATE SET
                KIND = excluded.KIND,
                XBINNING = excluded.XBINNING,
                EXPTIME = excluded.EXPTIME,
                CCD_TEMP = excluded.CCD_TEMP,
                FILTER_NAME = excluded.FILTER_NAME,
                GAIN = excluded.GAIN,
                NCOMBINE = excluded.NCOMBINE,
                REGISTERED_AT = excluded.REGISTERED_AT,
                ID_EQUIPMENTS = excluded.ID_EQUIPMENTS,
                ID_TELESCOPE = excluded.ID_TELESCOPE;
            """,
            (
                k,
                fp,
                int(xbinning),
                float(exptime),
                float(ccd_temp) if ccd_temp is not None and math.isfinite(float(ccd_temp)) else None,
                flt,
                int(gain),
                int(ncombine) if ncombine is not None else None,
                now,
                _eq,
                _tel,
            ),
        )
        self.conn.commit()

    def delete_calibration_library_entry_by_path(self, file_path: str | Path) -> int:
        """Remove registry row for a master file (after deleting the FITS on disk). Returns deleted row count."""
        fp = str(Path(file_path).resolve())
        cur = self.conn.execute("DELETE FROM CALIBRATION_LIBRARY WHERE FILE_PATH = ?", (fp,))
        self.conn.commit()
        return int(cur.rowcount or 0)

    def calibration_library_path_tag_map(self) -> dict[str, dict[str, Any]]:
        """``Path.resolve().casefold()`` -> ``{id_equipments, id_telescope, camera, telescope, ...}`` for UI."""
        self._ensure_calibration_library_table()
        cur = self.conn.execute(
            """
            SELECT c.FILE_PATH, c.ID_EQUIPMENTS, c.ID_TELESCOPE,
                   e.CAMERANAME, e.ALIAS AS EQ_ALIAS,
                   t.TELESCOPENAME, t.ALIAS AS TEL_ALIAS
            FROM CALIBRATION_LIBRARY c
            LEFT JOIN EQUIPMENTS e ON c.ID_EQUIPMENTS = e.ID
            LEFT JOIN TELESCOPE t ON c.ID_TELESCOPE = t.ID
            """
        )
        out: dict[str, dict[str, Any]] = {}
        for row in cur.fetchall():
            try:
                key = str(Path(str(row["FILE_PATH"])).resolve()).casefold()
            except OSError:
                key = str(row["FILE_PATH"]).casefold()
            out[key] = {
                "id_equipments": row["ID_EQUIPMENTS"],
                "id_telescope": row["ID_TELESCOPE"],
                "camera": row["CAMERANAME"],
                "eq_alias": row["EQ_ALIAS"],
                "telescope": row["TELESCOPENAME"],
                "tel_alias": row["TEL_ALIAS"],
            }
        return out

    def find_best_calibration_library_path(
        self,
        *,
        kind: str,
        xbinning: int,
        exptime: float,
        ccd_temp: float,
        filter_name: str = "",
        gain: int = 0,
        temp_tolerance: float = 0.5,
        prefer_unbinned_master: bool = True,
        id_equipments: int | None = None,
        id_telescope: int | None = None,
    ) -> str | None:
        """Return best existing file path for lights/calibration matching, or None.

        Prefers smallest |ΔT| among rows within ``temp_tolerance``, then newest file mtime.

        When ``prefer_unbinned_master`` is True and ``xbinning`` > 1, a **1×1** library row
        is tried first so the pipeline can resample the master to the light binning on-the-fly.

        When ``id_equipments`` and ``id_telescope`` are set, rows must either match that set
        exactly or be **legacy** (both IDs NULL = všeobecný master pre všetky sety).
        """
        k = str(kind or "").strip().lower()
        if k not in ("dark", "flat"):
            return None
        flt = "" if k == "dark" else str(filter_name or "").strip()
        _scope = (
            id_equipments is not None
            and id_telescope is not None
            and math.isfinite(float(int(id_equipments)))
            and math.isfinite(float(int(id_telescope)))
        )

        def _query_rows(xb: int) -> list[sqlite3.Row]:
            params: list[Any] = [
                k,
                int(xb),
                float(exptime),
                int(gain),
                flt,
                float(ccd_temp),
                float(temp_tolerance),
            ]
            scope_sql = ""
            if _scope:
                scope_sql = (
                    " AND ("
                    "(ID_EQUIPMENTS IS NULL AND ID_TELESCOPE IS NULL) "
                    "OR (ID_EQUIPMENTS = ? AND ID_TELESCOPE = ?)"
                    ")"
                )
                params.extend([int(id_equipments), int(id_telescope)])
            cur = self.conn.execute(
                f"""
                SELECT FILE_PATH, CCD_TEMP
                FROM CALIBRATION_LIBRARY
                WHERE KIND = ?
                  AND XBINNING = ?
                  AND EXPTIME = ?
                  AND COALESCE(GAIN, 0) = ?
                  AND (
                    (KIND = 'dark' AND COALESCE(FILTER_NAME, '') = '')
                    OR (KIND = 'flat' AND FILTER_NAME = ?)
                  )
                  AND (
                    CCD_TEMP IS NULL
                    OR ABS(CCD_TEMP - ?) <= ?
                  )
                  {scope_sql}
                ;
                """,
                tuple(params),
            )
            return cur.fetchall()

        def _score(rows_in: list[sqlite3.Row]) -> str | None:
            scored: list[tuple[float, float, str]] = []
            for row in rows_in:
                p = Path(str(row["FILE_PATH"]))
                if not p.is_file():
                    continue
                tdb = row["CCD_TEMP"]
                if tdb is None:
                    tdelta = 1e9
                else:
                    try:
                        tdelta = abs(float(tdb) - float(ccd_temp))
                    except (TypeError, ValueError):
                        tdelta = 1e9
                try:
                    mtime = -float(p.stat().st_mtime)
                except OSError:
                    mtime = 0.0
                scored.append((tdelta, mtime, str(p)))
            if not scored:
                return None
            scored.sort(key=lambda x: (x[0], x[1]))
            return scored[0][2]

        rows = _query_rows(int(xbinning))
        if prefer_unbinned_master and int(xbinning) > 1:
            rows_1 = _query_rows(1)
            hit1 = _score(rows_1)
            if hit1 is not None:
                return hit1
        hit = _score(rows)
        return hit

    def fetch_obs_draft_telescope_equipment(self, draft_id: int) -> dict[str, Any] | None:
        """JOIN OBS_DRAFT → TELESCOPE / EQUIPMENTS for UI (telescope + sensor summary)."""
        row = self.conn.execute(
            """
            SELECT
                d.ID AS draft_id,
                t.TELESCOPENAME AS telescope_name,
                t.FOCAL AS telescope_focal_mm,
                e.CAMERANAME AS equipment_name,
                e.PIXELSIZE AS pixel_um
            FROM OBS_DRAFT d
            LEFT JOIN TELESCOPE t ON d.ID_TELESCOPE = t.ID
            LEFT JOIN EQUIPMENTS e ON d.ID_EQUIPMENTS = e.ID
            WHERE d.ID = ?;
            """,
            (int(draft_id),),
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _normalize_calibration_library_filter_name(flt: str | None) -> str:
        s = str(flt or "").strip()
        if not s or s.lower() in {"unknown", "none", "nan"}:
            return "NoFilter"
        return s

    def calibration_library_has_flat_for_filter(self, filter_name: str | None) -> bool:
        """True if CALIBRATION_LIBRARY has at least one flat row for this filter name."""
        flt = self._normalize_calibration_library_filter_name(filter_name)
        row = self.conn.execute(
            """
            SELECT 1 FROM CALIBRATION_LIBRARY
            WHERE KIND = 'flat' AND FILTER_NAME = ?
            LIMIT 1;
            """,
            (flt,),
        ).fetchone()
        return row is not None

    def _ensure_obs_files_draft_column(self) -> None:
        """Migration: ensure OBS_FILES has DRAFT_ID column (older DBs)."""
        cursor = self.conn.execute("PRAGMA table_info('OBS_FILES');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "DRAFT_ID" in cols:
            return

        # Rebuild table to add DRAFT_ID + FK (SQLite limitations)
        self.conn.executescript(
            """
            ALTER TABLE OBS_FILES RENAME TO OBS_FILES_OLD;
            CREATE TABLE OBS_FILES (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                OBSERVATION_ID TEXT,
                DRAFT_ID INTEGER,
                FILE_PATH TEXT,
                IMAGETYP TEXT,
                FILTER TEXT,
                FOREIGN KEY (OBSERVATION_ID) REFERENCES OBSERVATION (ID) ON DELETE CASCADE,
                FOREIGN KEY (DRAFT_ID) REFERENCES OBS_DRAFT (ID) ON DELETE CASCADE
            );
            INSERT INTO OBS_FILES (ID, OBSERVATION_ID, DRAFT_ID, FILE_PATH, IMAGETYP, FILTER)
            SELECT ID, OBSERVATION_ID, NULL, FILE_PATH, IMAGETYP, FILTER FROM OBS_FILES_OLD;
            DROP TABLE OBS_FILES_OLD;
            """
        )
        self.conn.commit()

    def _ensure_obs_files_qc_columns(self) -> None:
        """Post-calibration QC metrics per light file (matched by archived raw ``FILE_PATH``)."""
        cursor = self.conn.execute("PRAGMA table_info('OBS_FILES');")
        cols = {row["name"] for row in cursor.fetchall()}
        specs = (
            ("QC_HFR", "REAL"),
            ("QC_STARS", "INTEGER"),
            ("QC_BACKGROUND", "REAL"),
            ("QC_BG_RMS", "REAL"),
            ("QC_PASSED", "INTEGER"),
        )
        for name, sql_type in specs:
            if name not in cols:
                self.conn.execute(f"ALTER TABLE OBS_FILES ADD COLUMN {name} {sql_type};")
        self.conn.commit()

    def _ensure_obs_files_quality_inspection_columns(self) -> None:
        """Quality inspection (DAO metrics, auto-reject, manual IS_REJECTED)."""
        cursor = self.conn.execute("PRAGMA table_info('OBS_FILES');")
        cols = {row["name"] for row in cursor.fetchall()}
        specs = (
            ("FWHM", "REAL"),
            ("SKY_LEVEL", "REAL"),
            ("STAR_COUNT", "INTEGER"),
            ("REJECTED_AUTO", "INTEGER"),
            ("IS_REJECTED", "INTEGER"),
            ("INSPECTION_JD", "REAL"),
            ("RA", "REAL"),
            ("DE", "REAL"),
            ("EXPTIME", "REAL"),
            ("DRIFT", "REAL"),
            ("DRIFT_DRA", "REAL"),
            ("DRIFT_DDE", "REAL"),
            ("ROUNDNESS_MEAN", "REAL"),
            ("ELONGATION_MEAN", "REAL"),
        )
        for name, sql_type in specs:
            if name not in cols:
                self.conn.execute(f"ALTER TABLE OBS_FILES ADD COLUMN {name} {sql_type};")
        self.conn.commit()

    def _ensure_obs_files_observation_group_key(self) -> None:
        """``(FILTER|EXPTIME|BINNING)`` subgroup for multi-observation imports."""
        cursor = self.conn.execute("PRAGMA table_info('OBS_FILES');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "OBSERVATION_GROUP_KEY" not in cols:
            self.conn.execute("ALTER TABLE OBS_FILES ADD COLUMN OBSERVATION_GROUP_KEY TEXT;")
        self.conn.commit()

    def _ensure_obs_files_scanning_and_calibration_columns(self) -> None:
        """Per-file scanning key + calibration mode metadata in OBS_FILES."""
        cursor = self.conn.execute("PRAGMA table_info('OBS_FILES');")
        cols = {row["name"] for row in cursor.fetchall()}
        specs = (
            ("ID_SCANNING", "INTEGER"),
            ("IS_CALIBRATED", "INTEGER"),
            ("CALIB_TYPE", "TEXT"),
            ("CALIB_FLAGS", "TEXT"),
        )
        for name, sql_type in specs:
            if name not in cols:
                self.conn.execute(f"ALTER TABLE OBS_FILES ADD COLUMN {name} {sql_type};")
        self.conn.commit()

    def _ensure_photometry_light_curve_table(self) -> None:
        """Multi-filter light-curve storage (JD vs magnitude), keyed by draft / filter."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS PHOTOMETRY_LIGHT_CURVE (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                DRAFT_ID INTEGER,
                OBSERVATION_ID TEXT,
                OBSERVATION_GROUP_KEY TEXT NOT NULL DEFAULT '',
                FILTER_NAME TEXT NOT NULL DEFAULT '',
                JD REAL NOT NULL,
                MAG REAL,
                MAG_ERR REAL,
                SOURCE_FILE TEXT,
                CREATED_AT TEXT NOT NULL,
                FOREIGN KEY (DRAFT_ID) REFERENCES OBS_DRAFT (ID) ON DELETE CASCADE
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_PHOT_LC_DRAFT_JD ON PHOTOMETRY_LIGHT_CURVE (DRAFT_ID, JD);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_PHOT_LC_FILTER ON PHOTOMETRY_LIGHT_CURVE (FILTER_NAME);"
        )
        self.conn.commit()

    def replace_photometry_light_curve_for_draft(
        self,
        draft_id: int,
        rows: list[dict[str, Any]],
        *,
        clear_existing: bool = True,
    ) -> int:
        """Replace (or append) photometry points for a draft. Rows: filter_name, jd, mag, mag_err?, group_key?, source_file?."""
        did = int(draft_id)
        if clear_existing:
            self.conn.execute("DELETE FROM PHOTOMETRY_LIGHT_CURVE WHERE DRAFT_ID = ?;", (did,))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n = 0
        for r in rows:
            try:
                jd = float(r.get("jd"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(jd):
                continue
            flt = str(r.get("filter_name") or r.get("filter") or "").strip() or "Unknown"
            gk = str(r.get("observation_group_key") or r.get("group_key") or "").strip()
            mag = r.get("mag")
            me = r.get("mag_err")
            mag_v = float(mag) if mag is not None and math.isfinite(float(mag)) else None
            err_v = float(me) if me is not None and math.isfinite(float(me)) else None
            src = str(r.get("source_file") or "")
            self.conn.execute(
                """
                INSERT INTO PHOTOMETRY_LIGHT_CURVE (
                    DRAFT_ID, OBSERVATION_ID, OBSERVATION_GROUP_KEY, FILTER_NAME, JD, MAG, MAG_ERR, SOURCE_FILE, CREATED_AT
                ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?);
                """,
                (did, gk, flt, jd, mag_v, err_v, src, now),
            )
            n += 1
        self.conn.commit()
        return n

    def fetch_photometry_light_curve_for_draft(self, draft_id: int) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT OBSERVATION_GROUP_KEY, FILTER_NAME, JD, MAG, MAG_ERR, SOURCE_FILE
            FROM PHOTOMETRY_LIGHT_CURVE
            WHERE DRAFT_ID = ?
            ORDER BY FILTER_NAME, JD;
            """,
            (int(draft_id),),
        )
        return [dict(row) for row in cur.fetchall()]

    def _ensure_scanning_gain_column(self) -> None:
        """Schema migration: add GAIN column to SCANNING."""
        cursor = self.conn.execute("PRAGMA table_info('SCANNING');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "GAIN" not in cols:
            self.conn.execute("ALTER TABLE SCANNING ADD COLUMN GAIN INTEGER DEFAULT 0;")
            self.conn.execute("UPDATE SCANNING SET GAIN = 0 WHERE GAIN IS NULL;")
            self.conn.commit()

    def _ensure_obs_draft_masterstar_path_column(self) -> None:
        """Schema migration: persist resolved MASTERSTAR path per draft."""
        cursor = self.conn.execute("PRAGMA table_info('OBS_DRAFT');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "MASTERSTAR_PATH" not in cols:
            self.conn.execute("ALTER TABLE OBS_DRAFT ADD COLUMN MASTERSTAR_PATH TEXT;")
        if "MASTERSTAR_FITS_PATH" not in cols:
            self.conn.execute("ALTER TABLE OBS_DRAFT ADD COLUMN MASTERSTAR_FITS_PATH TEXT;")
        self.conn.commit()

    def _ensure_obs_draft_status_panel_columns(self) -> None:
        """Schema migration: optional persisted status-panel optics values on draft."""
        cursor = self.conn.execute("PRAGMA table_info('OBS_DRAFT');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "FOCAL_MM" not in cols:
            self.conn.execute("ALTER TABLE OBS_DRAFT ADD COLUMN FOCAL_MM REAL;")
        if "PIXEL_UM" not in cols:
            self.conn.execute("ALTER TABLE OBS_DRAFT ADD COLUMN PIXEL_UM REAL;")
        self.conn.commit()

    def _ensure_active_columns(self) -> None:
        """Schema migration: add ACTIVE columns to EQUIPMENTS/TELESCOPE."""
        # EQUIPMENTS
        cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
        eq_cols = {row["name"] for row in cursor.fetchall()}
        if "ACTIVE" not in eq_cols:
            self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN ACTIVE TEXT DEFAULT 'YES';")
            self.conn.execute("UPDATE EQUIPMENTS SET ACTIVE = 'YES' WHERE ACTIVE IS NULL;")

        # TELESCOPE
        cursor = self.conn.execute("PRAGMA table_info('TELESCOPE');")
        tel_cols = {row["name"] for row in cursor.fetchall()}
        if "ACTIVE" not in tel_cols:
            self.conn.execute("ALTER TABLE TELESCOPE ADD COLUMN ACTIVE INTEGER DEFAULT 1;")
            self.conn.execute("UPDATE TELESCOPE SET ACTIVE = 1 WHERE ACTIVE IS NULL;")

        self._normalize_telescope_active_to_binary()
        self.conn.commit()

    def _normalize_telescope_active_to_binary(self) -> None:
        """Store ``TELESCOPE.ACTIVE`` only as **0** or **1** (legacy YES/NO/1 → 1, 0/NO → 0)."""
        cur = self.conn.execute("SELECT ID, ACTIVE FROM TELESCOPE;")
        rows = cur.fetchall()
        for row in rows:
            rid = int(row["ID"])
            nv = int(self.normalize_active_db_value(row["ACTIVE"]))
            self.conn.execute("UPDATE TELESCOPE SET ACTIVE = ? WHERE ID = ?;", (nv, rid))

    def _ensure_equipments_saturate_adu_column(self) -> None:
        """Schema migration: linear saturation ceiling (ADU) per camera/equipment row."""
        cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "SATURATE_ADU" in cols:
            return
        self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN SATURATE_ADU REAL;")
        self.conn.commit()

    def _ensure_equipments_cosmic_columns(self) -> None:
        """Optional detector params for L.A.Cosmic / astroscrappy (e⁻/ADU, read noise e⁻)."""
        cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "GAIN_ADU" not in cols:
            self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN GAIN_ADU REAL;")
        if "READNOISE_E" not in cols:
            self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN READNOISE_E REAL;")
        self.conn.commit()

    def _ensure_equipments_focal_column(self) -> None:
        """Optional focal length [mm] on EQUIPMENTS (per-camera plate-scale / diagnostics)."""
        cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "FOCAL" not in cols:
            self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN FOCAL REAL;")
            self.conn.commit()

    def _ensure_settings_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS SETTINGS (
                KEY TEXT PRIMARY KEY,
                VALUE TEXT
            );
            """
        )
        self.conn.commit()

    def _seed_default_settings(self) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO SETTINGS (KEY, VALUE) VALUES (?, ?);",
            ("masterdark_validity_days", "60"),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO SETTINGS (KEY, VALUE) VALUES (?, ?);",
            ("masterflat_validity_days", "200"),
        )
        self.conn.commit()

    def get_setting_int(self, key: str, default: int) -> int:
        cursor = self.conn.execute("SELECT VALUE FROM SETTINGS WHERE KEY = ?;", (key,))
        row = cursor.fetchone()
        if row is None:
            return default
        try:
            return int(row["VALUE"])
        except (TypeError, ValueError):
            return default

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO SETTINGS (KEY, VALUE) VALUES (?, ?) "
            "ON CONFLICT(KEY) DO UPDATE SET VALUE=excluded.VALUE;",
            (key, value),
        )
        self.conn.commit()

    def _ensure_observation_import_columns(self) -> None:
        """Lightweight schema migration for import logging fields."""
        cursor = self.conn.execute("PRAGMA table_info('OBSERVATION');")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        to_add: list[tuple[str, str]] = [
            ("LIGHTS_PATH", "TEXT"),
            ("CALIB_PATH", "TEXT"),
            ("IMPORTED_AT", "TEXT"),
            ("IMPORT_WARNINGS", "TEXT"),
            ("IS_CALIBRATED", "INTEGER"),
            ("ARCHIVE_PATH", "TEXT"),
        ]
        for col_name, col_type in to_add:
            if col_name in existing_cols:
                continue
            self.conn.execute(f"ALTER TABLE OBSERVATION ADD COLUMN {col_name} {col_type};")
        self.conn.commit()

    def initialize_database(self) -> None:
        """Populate default seed data for required reference tables."""
        db = self.conn

        # 1. Tabulka EQUIPMENTS
        db.execute(
            "INSERT OR IGNORE INTO EQUIPMENTS (ID, CAMERANAME, ALIAS, SENSORTYPE, SENSORSIZE, PIXELSIZE) "
            "VALUES (1, 'QHY294MM', 'Camera1', 'IMX492', '4164*2796', 4.63)"
        )
        # 2. Tabulka TELESCOPE
        db.execute(
            "INSERT OR IGNORE INTO TELESCOPE (ID, TELESCOPENAME, ALIAS, DIAMETER, FOCAL) "
            "VALUES (1, 'Carl-Zeiss', 'Teleobjektiv1', 72.0, 200.0)"
        )
        # 3. Tabulka LOCATION
        db.execute(
            "INSERT OR IGNORE INTO LOCATION (ID, PLACENAME, LATITUDE, LONGITUDE, ALTITUDE) "
            "VALUES (1, 'Dablice', 50.073658, 14.418540, 355.5)"
        )
        # 4. Tabulka SCANNING
        db.execute(
            "INSERT OR IGNORE INTO SCANNING (ID, EXPTIME, FILTERS, BINNING, SENSORTEMP) "
            "VALUES (1, 120.0, 'Clear', 11, -10.0)"
        )
        db.commit()

    def _ensure_obs_files_indexes(self) -> None:
        """Performance indexes for staging tables used heavily in the pipeline."""
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_OBS_FILES_DRAFT_ID ON OBS_FILES (DRAFT_ID);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_OBS_FILES_OBS_ID ON OBS_FILES (OBSERVATION_ID);"
        )
        self.conn.commit()

    @staticmethod
    def _jd_to_yyyymmdd(jd: float) -> str:
        """Convert Julian date to UTC date string YYYYMMDD.

        Robust against missing/invalid JD values (falls back to today's UTC date).
        """
        try:
            jd_f = float(jd)
        except (TypeError, ValueError):
            return datetime.now(timezone.utc).strftime("%Y%m%d")

        # JD sanity bounds (very permissive); guard against 0/NaN/inf.
        if (not math.isfinite(jd_f)) or jd_f < 2_000_000 or jd_f > 3_500_000:
            return datetime.now(timezone.utc).strftime("%Y%m%d")

        unix_seconds = (jd_f - 2440587.5) * 86400.0
        try:
            dt_utc = datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return datetime.now(timezone.utc).strftime("%Y%m%d")
        return dt_utc.strftime("%Y%m%d")

    @staticmethod
    def generate_hashtag(
        id_equipments: int,
        id_telescope: int,
        id_location: int,
        id_scanning: int,
        center_of_field_ra: float,
        center_of_field_de: float,
        observation_start_jd: float,
    ) -> str:
        """Create deterministic observation id in format YYYYMMDD_HASH."""
        payload = (
            f"{id_equipments}|{id_telescope}|{id_location}|{id_scanning}|"
            f"{center_of_field_ra:.8f}|{center_of_field_de:.8f}|{observation_start_jd:.8f}"
        )
        digest = hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:6]
        date_prefix = VyvarDatabase._jd_to_yyyymmdd(observation_start_jd)
        return f"{date_prefix}_{digest}"

    def insert_equipment(
        self,
        camera_name: str,
        alias: str,
        sensor_type: str,
        sensor_size: str,
        pixel_size: float,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO EQUIPMENTS (CAMERANAME, ALIAS, SENSORTYPE, SENSORSIZE, PIXELSIZE)
            VALUES (?, ?, ?, ?, ?);
            """,
            (camera_name, alias, sensor_type, sensor_size, pixel_size),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def get_equipment_saturation_adu(self, equipment_id: int) -> float | None:
        """Return EQUIPMENTS.SATURATE_ADU if set and positive; else None."""
        cursor = self.conn.execute(
            "SELECT SATURATE_ADU FROM EQUIPMENTS WHERE ID = ?;",
            (int(equipment_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        v = row["SATURATE_ADU"]
        if v is None:
            return None
        try:
            f = float(v)
            return f if f > 0 and math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def get_equipment_pixel_size_um(self, equipment_id: int) -> float | None:
        """``EQUIPMENTS.PIXELSIZE``: native (1×1) pixel pitch [µm], if set and positive.

        With ``TELESCOPE``/``EQUIPMENTS.FOCAL`` and X/Y binning from the FITS header, the pipeline derives
        ``expected_scale`` (arcsec/px) for plate solve and catalog geometry.
        """
        cursor = self.conn.execute(
            "SELECT PIXELSIZE FROM EQUIPMENTS WHERE ID = ?;",
            (int(equipment_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        v = row["PIXELSIZE"]
        if v is None:
            return None
        try:
            f = float(v)
            return f if f > 0 and math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def get_equipment_focal_mm(self, equipment_id: int) -> float | None:
        """``EQUIPMENTS.FOCAL`` [mm] when column exists and value is positive (optional per-camera override).

        With ``effective_pixel_um_plate_scale`` from FITS/DB metadata this yields the expected plate scale
        (e.g. ~9.55″/px for a typical 200 mm refractor and ~9.3 µm pixels), which ``pipeline`` uses for
        Astrometry.net bounds, VYVAR Gaia solving, optional CD rescaling on MASTERSTAR, and cone sizing.
        """
        try:
            cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
            cols = {row["name"] for row in cursor.fetchall()}
        except Exception:  # noqa: BLE001
            return None
        if "FOCAL" not in cols:
            return None
        cursor = self.conn.execute(
            "SELECT FOCAL FROM EQUIPMENTS WHERE ID = ?;",
            (int(equipment_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        v = row["FOCAL"]
        if v is None:
            return None
        try:
            f = float(v)
            return f if f > 0 and math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def get_equipment_cosmic_params(self, equipment_id: int) -> tuple[float | None, float | None]:
        """Return ``(GAIN_ADU, READNOISE_E)`` from EQUIPMENTS when set (positive, finite)."""
        cursor = self.conn.execute(
            "SELECT GAIN_ADU, READNOISE_E FROM EQUIPMENTS WHERE ID = ?;",
            (int(equipment_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None, None

        def _pos(x: Any) -> float | None:
            if x is None:
                return None
            try:
                f = float(x)
                return f if f > 0 and math.isfinite(f) else None
            except (TypeError, ValueError):
                return None

        return _pos(row["GAIN_ADU"]), _pos(row["READNOISE_E"])

    def get_telescope_focal_mm(self, telescope_id: int | None = None) -> float | None:
        """Positive ``TELESCOPE.FOCAL`` [mm]. If ``telescope_id`` is None, first active row with FOCAL > 0."""
        if telescope_id is not None:
            cursor = self.conn.execute(
                "SELECT FOCAL FROM TELESCOPE WHERE ID = ?;",
                (int(telescope_id),),
            )
        else:
            act = self.sql_expr_active_is_true("ACTIVE")
            cursor = self.conn.execute(
                f"""
                SELECT FOCAL FROM TELESCOPE
                WHERE {act}
                  AND FOCAL IS NOT NULL AND FOCAL > 0
                ORDER BY ID
                LIMIT 1;
                """
            )
        row = cursor.fetchone()
        if row is None:
            return None
        v = row["FOCAL"]
        if v is None:
            return None
        try:
            f = float(v)
            return f if f > 0 and math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def get_combined_metadata(self, file_path: str | Path, draft_id: int) -> dict[str, Any]:
        """Merge FITS primary header with ``OBS_DRAFT`` → ``TELESCOPE`` / ``EQUIPMENTS`` SQL fallbacks.

        - Focal: plausible FITS keywords first; else ``EQUIPMENTS.FOCAL`` for the draft camera; else
          ``TELESCOPE.FOCAL`` via ``SELECT t.FOCAL FROM TELESCOPE t JOIN OBS_DRAFT d ON d.ID_TELESCOPE = t.ID WHERE d.ID = ?``.
        - Native pixel [µm]: FITS ``PIXSIZE*`` / ``XPIXSZ`` …; else ``EQUIPMENTS.PIXELSIZE`` via draft join
          (same as ``get_equipment_pixel_size_um`` on ``OBS_DRAFT.ID_EQUIPMENTS``).
        - **Binning:** ``XBINNING`` from FITS only; if missing → 1. Effective pixel for plate scale is
          ``native_pixel_um * XBINNING``.
        - ``SATURATE_ADU`` from ``EQUIPMENTS`` for the draft camera when set.

        Schema note: this project uses tables ``TELESCOPE`` / ``FOCAL`` and ``EQUIPMENTS`` / ``PIXELSIZE``,
        not ``TELESCOPES`` / ``FOCAL_LENGTH_MM`` / ``PIXEL_UM``.
        """
        fp = Path(file_path)
        did = int(draft_id)
        cur = self.conn.execute(
            "SELECT ID_EQUIPMENTS, ID_TELESCOPE FROM OBS_DRAFT WHERE ID = ?;",
            (did,),
        )
        dr = cur.fetchone()
        if dr is None:
            raise ValueError(f"OBS_DRAFT id={did} not found.")

        id_eq = dr["ID_EQUIPMENTS"]
        id_tel = dr["ID_TELESCOPE"]

        with fits.open(fp, memmap=False) as hdul:
            header = hdul[0].header.copy()

        x_bin = _db_xbinning_strict(header)
        y_bin = _db_ybinning_header(header, x_bin)

        focal_mm = _db_header_focal_length_mm(header)
        _f_hdr_ok = focal_mm is not None and _db_focal_plausible_mm(float(focal_mm))
        focal_src = "fits_header" if _f_hdr_ok else "none"
        if not _f_hdr_ok:
            focal_mm = None
            if id_eq is not None:
                try:
                    ef = self.get_equipment_focal_mm(int(id_eq))
                except Exception:  # noqa: BLE001
                    ef = None
                if ef is not None:
                    n, _ = normalize_telescope_focal_mm_for_plate_scale(float(ef))
                    if _db_focal_plausible_mm(n):
                        focal_mm = float(n)
                        focal_src = "equipment_focal_sql"
            if focal_mm is None and id_tel is not None:
                cur_tf = self.conn.execute(
                    """
                    SELECT t.FOCAL FROM TELESCOPE t
                    INNER JOIN OBS_DRAFT d ON d.ID_TELESCOPE = t.ID
                    WHERE d.ID = ?;
                    """,
                    (did,),
                )
                row_tf = cur_tf.fetchone()
                if row_tf is not None and row_tf["FOCAL"] is not None:
                    try:
                        raw_t = float(row_tf["FOCAL"])
                        n2, _ = normalize_telescope_focal_mm_for_plate_scale(raw_t)
                        if _db_focal_plausible_mm(n2):
                            focal_mm = float(n2)
                            focal_src = "telescope_focal_sql"
                    except (TypeError, ValueError):
                        pass

        native_um = _db_header_pixel_native_um_mean(header)
        pix_src = "fits_header"
        if native_um is None or not math.isfinite(native_um) or native_um <= 0 or native_um > 300.0:
            native_um = None
            pix_src = "none"
            if id_eq is not None:
                try:
                    native_um = self.get_equipment_pixel_size_um(int(id_eq))
                except Exception:  # noqa: BLE001
                    native_um = None
                if native_um is not None:
                    pix_src = "equipment_pixelsize_sql"

        pixel_effective_um: float | None = None
        if native_um is not None and math.isfinite(float(native_um)) and float(native_um) > 0:
            pixel_effective_um = float(native_um) * float(x_bin)

        sat: float | None = None
        if id_eq is not None:
            try:
                sat = self.get_equipment_saturation_adu(int(id_eq))
            except Exception:  # noqa: BLE001
                sat = None

        if focal_mm is not None:
            n_f, _ = normalize_telescope_focal_mm_for_plate_scale(float(focal_mm))
            if _db_focal_plausible_mm(n_f):
                focal_mm = float(n_f)

        return {
            "draft_id": did,
            "id_equipments": None if id_eq is None else int(id_eq),
            "id_telescope": None if id_tel is None else int(id_tel),
            "xbinning": int(x_bin),
            "ybinning": int(y_bin),
            "focal_length_mm": focal_mm,
            "focal_source": focal_src,
            "pixel_native_um": float(native_um) if native_um is not None else None,
            "pixel_effective_um": pixel_effective_um,
            "pixel_source": pix_src,
            "saturate_adu": sat,
        }

    def insert_telescope(
        self,
        telescope_name: str,
        alias: str,
        diameter: float,
        focal: float,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO TELESCOPE (TELESCOPENAME, ALIAS, DIAMETER, FOCAL)
            VALUES (?, ?, ?, ?);
            """,
            (telescope_name, alias, diameter, focal),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def insert_location(
        self,
        place_name: str,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO LOCATION (PLACENAME, LATITUDE, LONGITUDE, ALTITUDE)
            VALUES (?, ?, ?, ?);
            """,
            (place_name, latitude, longitude, altitude),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def insert_scanning(
        self,
        exp_time: float,
        filters: str,
        binning: int,
        sensor_temp: float,
        gain: int = 0,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO SCANNING (EXPTIME, FILTERS, BINNING, SENSORTEMP, GAIN)
            VALUES (?, ?, ?, ?, ?);
            """,
            (exp_time, filters, binning, sensor_temp, int(gain)),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def find_or_create_scanning_id(self, metadata: dict[str, Any]) -> int:
        """Find scanning row by metadata, or create a new one.

        Matching fields:
        - EXPTIME exact
        - FILTERS exact
        - BINNING exact
        - SENSORTEMP with tolerance +/- 0.5
        """
        exp_time = float(metadata["exposure"])
        filters = str(metadata["filter"])
        binning = int(metadata["binning"])
        sensor_temp = float(metadata["temp"])
        gain = int(metadata.get("gain", 0))

        cursor = self.conn.execute(
            """
            SELECT ID
            FROM SCANNING
            WHERE EXPTIME = ?
              AND FILTERS = ?
              AND BINNING = ?
              AND ABS(SENSORTEMP - ?) <= 0.5
              AND COALESCE(GAIN, 0) = ?
            ORDER BY ID
            LIMIT 1;
            """,
            (exp_time, filters, binning, sensor_temp, gain),
        )
        row = cursor.fetchone()
        if row is not None:
            return int(row["ID"])

        return self.insert_scanning(
            exp_time=exp_time,
            filters=filters,
            binning=binning,
            sensor_temp=sensor_temp,
            gain=gain,
        )

    def insert_observation(
        self,
        id_equipments: int,
        id_telescope: int,
        id_location: int,
        id_scanning: int,
        center_of_field_ra: float,
        center_of_field_de: float,
        observation_start_jd: float,
        hashtag: str | None = None,
    ) -> str:
        observation_id = hashtag or self.generate_hashtag(
            id_equipments=id_equipments,
            id_telescope=id_telescope,
            id_location=id_location,
            id_scanning=id_scanning,
            center_of_field_ra=center_of_field_ra,
            center_of_field_de=center_of_field_de,
            observation_start_jd=observation_start_jd,
        )

        try:
            self.conn.execute(
                """
                INSERT INTO OBSERVATION (
                    ID,
                    ID_EQUIPMENTS,
                    ID_TELESCOPE,
                    ID_LOCATION,
                    ID_SCANNING,
                    CENTEROFFIELDRA,
                    CENTEROFFIELDDE,
                    OBSERVATIONSTARTJD
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    observation_id,
                    id_equipments,
                    id_telescope,
                    id_location,
                    id_scanning,
                    center_of_field_ra,
                    center_of_field_de,
                    observation_start_jd,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "FOREIGN KEY constraint failed" in str(exc):
                raise ValueError(
                    "Observation references a non-existing foreign key "
                    "(equipment, telescope, location, or scanning)."
                ) from exc
            if "UNIQUE constraint failed" in str(exc):
                raise ValueError(
                    f"Observation with hashtag '{observation_id}' already exists."
                ) from exc
            raise

        self.conn.commit()
        return observation_id

    def add_observation(self, data_dict: dict[str, Any]) -> str:
        """Insert observation from input dictionary and auto-generate hashtag ID."""
        required_fields = {
            "center_of_field_ra",
            "center_of_field_de",
            "observation_start_jd",
        }
        missing = required_fields - set(data_dict.keys())
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(f"Missing required observation fields: {missing_fields}")

        return self.insert_observation(
            id_equipments=int(data_dict.get("id_equipments", 1)),
            id_telescope=int(data_dict.get("id_telescope", 1)),
            id_location=int(data_dict.get("id_location", 1)),
            id_scanning=int(data_dict.get("id_scanning", 1)),
            center_of_field_ra=float(data_dict["center_of_field_ra"]),
            center_of_field_de=float(data_dict["center_of_field_de"]),
            observation_start_jd=float(data_dict["observation_start_jd"]),
            hashtag=None,
        )

    def get_equipments(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        sql = """
            SELECT ID, CAMERANAME, ALIAS, ACTIVE
            FROM EQUIPMENTS
        """
        params: tuple[object, ...] = ()
        if active_only:
            sql += f" WHERE {self.sql_expr_active_is_true('ACTIVE')} "
        sql += " ORDER BY ID; "
        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_telescopes(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        sql = """
            SELECT ID, TELESCOPENAME, ALIAS, ACTIVE
            FROM TELESCOPE
        """
        params: tuple[object, ...] = ()
        if active_only:
            sql += f" WHERE {self.sql_expr_active_is_true('ACTIVE')} "
        sql += " ORDER BY ID; "
        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_observation_metadata(self, hashtag: str) -> dict[str, Any] | None:
        cursor = self.conn.execute(
            """
            SELECT
                o.ID AS observation_id,
                o.CENTEROFFIELDRA AS center_of_field_ra,
                o.CENTEROFFIELDDE AS center_of_field_de,
                o.OBSERVATIONSTARTJD AS observation_start_jd,
                e.ID AS equipment_id,
                e.CAMERANAME AS camera_name,
                e.ALIAS AS equipment_alias,
                e.SENSORTYPE AS sensor_type,
                e.SENSORSIZE AS sensor_size,
                e.PIXELSIZE AS pixel_size,
                t.ID AS telescope_id,
                t.TELESCOPENAME AS telescope_name,
                t.ALIAS AS telescope_alias,
                t.DIAMETER AS diameter,
                t.FOCAL AS focal,
                l.ID AS location_id,
                l.PLACENAME AS place_name,
                l.LATITUDE AS latitude,
                l.LONGITUDE AS longitude,
                l.ALTITUDE AS altitude,
                s.ID AS scanning_id,
                s.EXPTIME AS exp_time,
                s.FILTERS AS filters,
                s.BINNING AS binning,
                s.SENSORTEMP AS sensor_temp
            FROM OBSERVATION o
            LEFT JOIN EQUIPMENTS e ON o.ID_EQUIPMENTS = e.ID
            LEFT JOIN TELESCOPE t ON o.ID_TELESCOPE = t.ID
            LEFT JOIN LOCATION l ON o.ID_LOCATION = l.ID
            LEFT JOIN SCANNING s ON o.ID_SCANNING = s.ID
            WHERE o.ID = ?;
            """,
            (hashtag,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def update_observation_import_log(
        self,
        hashtag: str,
        *,
        lights_path: str,
        calib_path: str,
        imported_at: str,
        import_warnings: str | None = None,
        is_calibrated: bool | None = None,
        archive_path: str | None = None,
    ) -> None:
        is_cal_int = None if is_calibrated is None else (1 if is_calibrated else 0)
        cursor = self.conn.execute(
            """
            UPDATE OBSERVATION
            SET LIGHTS_PATH = ?,
                CALIB_PATH = ?,
                IMPORTED_AT = ?,
                IMPORT_WARNINGS = ?,
                IS_CALIBRATED = COALESCE(?, IS_CALIBRATED),
                ARCHIVE_PATH = COALESCE(?, ARCHIVE_PATH)
            WHERE ID = ?;
            """,
            (
                lights_path,
                calib_path,
                imported_at,
                import_warnings,
                is_cal_int,
                archive_path,
                hashtag,
            ),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Observation '{hashtag}' not found for import log update.")
        self.conn.commit()

    def insert_observation_files(self, observation_id: str, files: list[dict[str, Any]]) -> None:
        """Insert per-file evidence rows into OBS_FILES for finalized OBSERVATION."""
        self.conn.execute("DELETE FROM OBS_FILES WHERE OBSERVATION_ID = ?;", (observation_id,))
        self.conn.executemany(
            """
            INSERT INTO OBS_FILES (
                OBSERVATION_ID, DRAFT_ID, FILE_PATH, IMAGETYP, FILTER, OBSERVATION_GROUP_KEY,
                ID_SCANNING, IS_CALIBRATED, CALIB_TYPE, CALIB_FLAGS
            )
            VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    observation_id,
                    str(item.get("file_path", "")),
                    str(item.get("imagetyp", "")),
                    str(item.get("filter", "")),
                    str(item.get("observation_group_key", "") or ""),
                    (int(item.get("id_scanning")) if item.get("id_scanning") is not None else None),
                    (int(item.get("is_calibrated")) if item.get("is_calibrated") is not None else None),
                    str(item.get("calib_type", "") or ""),
                    str(item.get("calib_flags", "") or ""),
                )
                for item in files
            ],
        )
        self.conn.commit()

    def count_obs_files_for_observation(self, observation_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM OBS_FILES WHERE OBSERVATION_ID = ?;",
            (observation_id,),
        ).fetchone()
        return int(row["n"] if row is not None and row["n"] is not None else 0)

    def count_obs_files_for_draft(self, draft_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM OBS_FILES WHERE DRAFT_ID = ?;",
            (int(draft_id),),
        ).fetchone()
        return int(row["n"] if row is not None and row["n"] is not None else 0)

    def insert_draft_files(self, draft_id: int, files: list[dict[str, Any]]) -> None:
        """Insert per-file evidence rows into OBS_FILES for draft ingestion."""
        self.conn.execute("DELETE FROM OBS_FILES WHERE DRAFT_ID = ?;", (int(draft_id),))
        self.conn.executemany(
            """
            INSERT INTO OBS_FILES (
                OBSERVATION_ID, DRAFT_ID, FILE_PATH, IMAGETYP, FILTER, OBSERVATION_GROUP_KEY,
                ID_SCANNING, IS_CALIBRATED, CALIB_TYPE, CALIB_FLAGS
            )
            VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    int(draft_id),
                    str(item.get("file_path", "")),
                    str(item.get("imagetyp", "")),
                    str(item.get("filter", "")),
                    str(item.get("observation_group_key", "") or ""),
                    (int(item.get("id_scanning")) if item.get("id_scanning") is not None else None),
                    (int(item.get("is_calibrated")) if item.get("is_calibrated") is not None else None),
                    str(item.get("calib_type", "") or ""),
                    str(item.get("calib_flags", "") or ""),
                )
                for item in files
            ],
        )
        self.conn.commit()

    def update_obs_file_qc_by_raw_light_path(
        self,
        raw_light_path: str | Path,
        *,
        draft_id: int | None = None,
        observation_id: str | None = None,
        qc_hfr: float | None = None,
        qc_stars: int | None = None,
        qc_background: float | None = None,
        qc_bg_rms: float | None = None,
        qc_passed: bool | None = None,
    ) -> int:
        """Update QC columns for the row whose ``FILE_PATH`` matches the raw light FITS path.

        Restrict with ``draft_id`` or ``observation_id`` when set to avoid ambiguous basename matches.
        Returns SQLite ``rowcount``.
        """
        p = str(Path(raw_light_path).resolve())
        assignments: list[str] = []
        set_vals: list[Any] = []
        if qc_hfr is not None and math.isfinite(float(qc_hfr)):
            assignments.append("QC_HFR = ?")
            set_vals.append(float(qc_hfr))
        if qc_stars is not None:
            assignments.append("QC_STARS = ?")
            set_vals.append(int(qc_stars))
        if qc_background is not None and math.isfinite(float(qc_background)):
            assignments.append("QC_BACKGROUND = ?")
            set_vals.append(float(qc_background))
        if qc_bg_rms is not None and math.isfinite(float(qc_bg_rms)):
            assignments.append("QC_BG_RMS = ?")
            set_vals.append(float(qc_bg_rms))
        if qc_passed is not None:
            assignments.append("QC_PASSED = ?")
            set_vals.append(1 if qc_passed else 0)
        if not assignments:
            return 0

        where_parts = ["FILE_PATH = ?"]
        where_vals: list[Any] = [p]
        if draft_id is not None:
            where_parts.append("DRAFT_ID = ?")
            where_vals.append(int(draft_id))
        elif observation_id is not None:
            where_parts.append("OBSERVATION_ID = ?")
            where_vals.append(str(observation_id))

        wh = " AND ".join(where_parts)
        sql = f"UPDATE OBS_FILES SET {', '.join(assignments)} WHERE {wh}"
        cur = self.conn.execute(sql, tuple(set_vals + where_vals))
        n = int(cur.rowcount or 0)
        if n == 0 and os.name == "nt":
            wparts = ["LOWER(FILE_PATH) = LOWER(?)"]
            wvals: list[Any] = [p]
            if draft_id is not None:
                wparts.insert(0, "DRAFT_ID = ?")
                wvals.insert(0, int(draft_id))
            elif observation_id is not None:
                wparts.insert(0, "OBSERVATION_ID = ?")
                wvals.insert(0, str(observation_id))
            cur2 = self.conn.execute(
                "SELECT ID FROM OBS_FILES WHERE " + " AND ".join(wparts),
                tuple(wvals),
            )
            row = cur2.fetchone()
            if row is None:
                self.conn.commit()
                return 0
            sql2 = f"UPDATE OBS_FILES SET {', '.join(assignments)} WHERE ID = ?"
            cur3 = self.conn.execute(sql2, tuple(set_vals + [int(row["ID"])]))
            n = int(cur3.rowcount or 0)
        self.conn.commit()
        return n

    def fetch_draft_light_rows_for_quality(self, draft_id: int) -> list[dict[str, Any]]:
        """Light frames for a draft (for quality inspection / preprocessing filter)."""
        cur = self.conn.execute(
            """
            SELECT ID, FILE_PATH, IMAGETYP, FILTER, OBSERVATION_GROUP_KEY, ID_SCANNING, IS_CALIBRATED, CALIB_TYPE,
                   CALIB_FLAGS,
                   FWHM, SKY_LEVEL, STAR_COUNT, REJECTED_AUTO, IS_REJECTED, INSPECTION_JD, RA, DE, EXPTIME, DRIFT,
                   DRIFT_DRA, DRIFT_DDE, ROUNDNESS_MEAN, ELONGATION_MEAN
            FROM OBS_FILES
            WHERE DRAFT_ID = ? AND LOWER(COALESCE(IMAGETYP, '')) = 'light'
            ORDER BY FILE_PATH;
            """,
            (int(draft_id),),
        )
        return [dict(r) for r in cur.fetchall()]

    def fetch_draft_scanning_ids(self, draft_id: int) -> list[int]:
        """Unique per-file scanning IDs for light rows in a draft."""
        cur = self.conn.execute(
            """
            SELECT DISTINCT ID_SCANNING
            FROM OBS_FILES
            WHERE DRAFT_ID = ? AND LOWER(COALESCE(IMAGETYP, '')) = 'light' AND ID_SCANNING IS NOT NULL
            ORDER BY ID_SCANNING;
            """,
            (int(draft_id),),
        )
        return [int(r["ID_SCANNING"]) for r in cur.fetchall() if r["ID_SCANNING"] is not None]

    def update_obs_file_calibration_state_by_raw_light_path(
        self,
        raw_light_path: str | Path,
        *,
        draft_id: int | None = None,
        observation_id: str | None = None,
        is_calibrated: int | None = None,
        calib_type: str | None = None,
        calib_flags: str | None = None,
    ) -> int:
        """Update IS_CALIBRATED/CALIB_TYPE/CALIB_FLAGS for one OBS_FILES row matched by FILE_PATH."""
        p = str(Path(raw_light_path).resolve())
        parts: list[str] = []
        vals: list[Any] = []
        if is_calibrated is not None:
            parts.append("IS_CALIBRATED = ?")
            vals.append(1 if int(is_calibrated) else 0)
        if calib_type is not None:
            parts.append("CALIB_TYPE = ?")
            vals.append(str(calib_type))
        if calib_flags is not None:
            parts.append("CALIB_FLAGS = ?")
            vals.append(str(calib_flags))
        if not parts:
            return 0
        where_parts = ["FILE_PATH = ?"]
        where_vals: list[Any] = [p]
        if draft_id is not None:
            where_parts.append("DRAFT_ID = ?")
            where_vals.append(int(draft_id))
        elif observation_id is not None:
            where_parts.append("OBSERVATION_ID = ?")
            where_vals.append(str(observation_id))
        sql = f"UPDATE OBS_FILES SET {', '.join(parts)} WHERE {' AND '.join(where_parts)}"
        cur = self.conn.execute(sql, tuple(vals + where_vals))
        self.conn.commit()
        return int(cur.rowcount or 0)

    def update_obs_file_quality_by_id(
        self,
        row_id: int,
        *,
        fwhm: float | None = None,
        sky_level: float | None = None,
        star_count: int | None = None,
        rejected_auto: int | None = None,
        inspection_jd: float | None = None,
        is_rejected: int | None = None,
        ra_deg: float | None = None,
        de_deg: float | None = None,
        exptime_sec: float | None = None,
        drift_arcmin: float | None = None,
        clear_drift: bool = False,
        drift_dra_deg: float | None = None,
        drift_dde_deg: float | None = None,
        roundness_mean: float | None = None,
        elongation_mean: float | None = None,
    ) -> None:
        """Update quality-inspection columns for one OBS_FILES row."""
        parts: list[str] = []
        vals: list[Any] = []
        if clear_drift:
            parts.extend(["DRIFT = NULL", "DRIFT_DRA = NULL", "DRIFT_DDE = NULL"])
        else:
            if drift_arcmin is not None and math.isfinite(float(drift_arcmin)) and float(drift_arcmin) >= 0.0:
                parts.append("DRIFT = ?")
                vals.append(float(drift_arcmin))
            if drift_dra_deg is not None and math.isfinite(float(drift_dra_deg)):
                parts.append("DRIFT_DRA = ?")
                vals.append(float(drift_dra_deg))
            if drift_dde_deg is not None and math.isfinite(float(drift_dde_deg)):
                parts.append("DRIFT_DDE = ?")
                vals.append(float(drift_dde_deg))
        if fwhm is not None and math.isfinite(float(fwhm)):
            parts.append("FWHM = ?")
            vals.append(float(fwhm))
        if sky_level is not None and math.isfinite(float(sky_level)):
            parts.append("SKY_LEVEL = ?")
            vals.append(float(sky_level))
        if star_count is not None:
            parts.append("STAR_COUNT = ?")
            vals.append(int(star_count))
        if rejected_auto is not None:
            parts.append("REJECTED_AUTO = ?")
            vals.append(1 if int(rejected_auto) else 0)
        if inspection_jd is not None and math.isfinite(float(inspection_jd)):
            parts.append("INSPECTION_JD = ?")
            vals.append(float(inspection_jd))
        if is_rejected is not None:
            parts.append("IS_REJECTED = ?")
            vals.append(1 if int(is_rejected) else 0)
        if ra_deg is not None and math.isfinite(float(ra_deg)):
            parts.append("RA = ?")
            vals.append(float(ra_deg))
        if de_deg is not None and math.isfinite(float(de_deg)):
            parts.append("DE = ?")
            vals.append(float(de_deg))
        if exptime_sec is not None and math.isfinite(float(exptime_sec)) and float(exptime_sec) >= 0:
            parts.append("EXPTIME = ?")
            vals.append(float(exptime_sec))
        if roundness_mean is not None and math.isfinite(float(roundness_mean)) and float(roundness_mean) >= 0.0:
            parts.append("ROUNDNESS_MEAN = ?")
            vals.append(float(roundness_mean))
        if elongation_mean is not None and math.isfinite(float(elongation_mean)) and float(elongation_mean) > 0.0:
            parts.append("ELONGATION_MEAN = ?")
            vals.append(float(elongation_mean))
        if not parts:
            return
        vals.append(int(row_id))
        self.conn.execute(
            f"UPDATE OBS_FILES SET {', '.join(parts)} WHERE ID = ?;",
            tuple(vals),
        )
        self.conn.commit()

    def bulk_update_obs_file_is_rejected(
        self,
        draft_id: int,
        updates: list[tuple[int, int]],
    ) -> None:
        """``updates`` = list of (row_id, is_rejected 0/1).

        Uses a short-lived connection so this stays valid when Streamlit
        ``data_editor`` ``on_change`` runs on a worker thread (not the thread
        that created ``self.conn``).
        """
        did = int(draft_id)
        con = sqlite3.connect(self.db_path)
        try:
            for rid, rej in updates:
                con.execute(
                    "UPDATE OBS_FILES SET IS_REJECTED = ? WHERE ID = ? AND DRAFT_ID = ?;",
                    (1 if int(rej) else 0, int(rid), did),
                )
            con.commit()
        finally:
            con.close()

    def _normalize_obs_file_path_key(self, p: str | Path) -> str:
        try:
            return str(Path(p).resolve()).casefold()
        except OSError:
            return str(p).casefold()

    def fetch_light_file_paths_not_rejected_for_draft(self, draft_id: int) -> set[str]:
        """Normalized ``FILE_PATH`` keys for draft lights with ``IS_REJECTED`` 0 or NULL."""
        cur = self.conn.execute(
            """
            SELECT FILE_PATH FROM OBS_FILES
            WHERE DRAFT_ID = ? AND LOWER(COALESCE(IMAGETYP, '')) = 'light'
              AND (IS_REJECTED IS NULL OR IS_REJECTED = 0);
            """,
            (int(draft_id),),
        )
        out: set[str] = set()
        for row in cur.fetchall():
            fp = row["FILE_PATH"]
            if fp:
                out.add(self._normalize_obs_file_path_key(str(fp)))
        return out

    def fetch_light_file_paths_not_rejected_for_observation(self, observation_id: str) -> set[str]:
        """Same as draft helper for finalized ``OBSERVATION_ID``."""
        cur = self.conn.execute(
            """
            SELECT FILE_PATH FROM OBS_FILES
            WHERE OBSERVATION_ID = ? AND LOWER(COALESCE(IMAGETYP, '')) = 'light'
              AND (IS_REJECTED IS NULL OR IS_REJECTED = 0);
            """,
            (str(observation_id),),
        )
        out: set[str] = set()
        for row in cur.fetchall():
            fp = row["FILE_PATH"]
            if fp:
                out.add(self._normalize_obs_file_path_key(str(fp)))
        return out

    def create_draft(self, data: dict[str, Any]) -> int:
        """Create an ingestion draft without Session ID (no astrometry yet)."""
        cursor = self.conn.execute(
            """
            INSERT INTO OBS_DRAFT (
                ID_EQUIPMENTS, ID_TELESCOPE, ID_LOCATION, ID_SCANNING,
                OBSERVATIONSTARTJD, CENTEROFFIELDRA, CENTEROFFIELDDE,
                STATUS, IS_CALIBRATED
            )
            VALUES (?, ?, ?, ?, ?, NULL, NULL, 'INGESTED', ?);
            """,
            (
                int(data.get("id_equipments", 1)),
                int(data.get("id_telescope", 1)),
                int(data.get("id_location", 1)),
                int(data.get("id_scanning", 1)),
                float(data.get("observation_start_jd", 0.0)),
                1 if bool(data.get("is_calibrated", False)) else 0,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def update_draft_import_log(
        self,
        draft_id: int,
        *,
        lights_path: str,
        calib_path: str,
        imported_at: str,
        import_warnings: str | None = None,
        is_calibrated: bool | None = None,
        archive_path: str | None = None,
    ) -> None:
        is_cal_int = None if is_calibrated is None else (1 if is_calibrated else 0)
        cursor = self.conn.execute(
            """
            UPDATE OBS_DRAFT
            SET LIGHTS_PATH = ?,
                CALIB_PATH = ?,
                IMPORTED_AT = ?,
                IMPORT_WARNINGS = ?,
                IS_CALIBRATED = COALESCE(?, IS_CALIBRATED),
                ARCHIVE_PATH = COALESCE(?, ARCHIVE_PATH)
            WHERE ID = ?;
            """,
            (
                lights_path,
                calib_path,
                imported_at,
                import_warnings,
                is_cal_int,
                archive_path,
                int(draft_id),
            ),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Draft '{draft_id}' not found for import log update.")
        self.conn.commit()

    def finalize_draft(self, draft_id: int, *, ra_deg: float, dec_deg: float) -> str:
        """Finalize a draft by creating OBSERVATION.ID after astrometry."""
        cursor = self.conn.execute("SELECT * FROM OBS_DRAFT WHERE ID = ?;", (int(draft_id),))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Draft '{draft_id}' not found.")

        observation_id = self.generate_hashtag(
            id_equipments=int(row["ID_EQUIPMENTS"]),
            id_telescope=int(row["ID_TELESCOPE"]),
            id_location=int(row["ID_LOCATION"]),
            id_scanning=int(row["ID_SCANNING"]),
            center_of_field_ra=float(ra_deg),
            center_of_field_de=float(dec_deg),
            observation_start_jd=float(row["OBSERVATIONSTARTJD"] or 0.0),
        )

        # Create OBSERVATION
        self.insert_observation(
            id_equipments=int(row["ID_EQUIPMENTS"]),
            id_telescope=int(row["ID_TELESCOPE"]),
            id_location=int(row["ID_LOCATION"]),
            id_scanning=int(row["ID_SCANNING"]),
            center_of_field_ra=float(ra_deg),
            center_of_field_de=float(dec_deg),
            observation_start_jd=float(row["OBSERVATIONSTARTJD"] or 0.0),
            hashtag=observation_id,
        )

        # Move file evidence to OBSERVATION
        self.conn.execute(
            """
            UPDATE OBS_FILES
            SET OBSERVATION_ID = ?, DRAFT_ID = NULL
            WHERE DRAFT_ID = ?;
            """,
            (observation_id, int(draft_id)),
        )

        # Mark draft finalized
        self.conn.execute(
            """
            UPDATE OBS_DRAFT
            SET STATUS = 'FINALIZED',
                FINAL_OBSERVATION_ID = ?,
                CENTEROFFIELDRA = ?,
                CENTEROFFIELDDE = ?
            WHERE ID = ?;
            """,
            (observation_id, float(ra_deg), float(dec_deg), int(draft_id)),
        )
        self.conn.commit()
        return observation_id

    def finalize_draft_to_observation(
        self,
        draft_id: int,
        *,
        approved_by: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Persist UI approval: upsert ``OBSERVATION`` row keyed by ``DRAFT_ID``, mark draft ``FINALIZED``.

        Returns the observation hashtag ``OBSERVATION.ID`` (text primary key).
        """
        from astropy.time import Time

        def _obs_cols() -> set[str]:
            cur = self.conn.execute("PRAGMA table_info(OBSERVATION);")
            return {str(r[1]) for r in cur.fetchall()}

        cols_o = _obs_cols()
        for col_sql in (
            ("DRAFT_ID", "INTEGER"),
            ("STATUS", "TEXT"),
            ("APPROVED_BY", "TEXT"),
            ("APPROVAL_NOTES", "TEXT"),
            ("APPROVAL_JD", "REAL"),
        ):
            name, typ = col_sql
            if name not in cols_o:
                self.conn.execute(f"ALTER TABLE OBSERVATION ADD COLUMN {name} {typ};")
        cols_o = _obs_cols()
        if "DRAFT_ID" in cols_o:
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS UQ_OBSERVATION_DRAFT_ID ON OBSERVATION(DRAFT_ID);"
            )

        cur_d = self.conn.execute("SELECT * FROM OBS_DRAFT WHERE ID = ?;", (int(draft_id),))
        drow = cur_d.fetchone()
        if drow is None:
            raise ValueError(f"Draft {draft_id} not found")
        draft = dict(drow)
        st = str(draft.get("STATUS") or "").strip().upper()
        if st == "FINALIZED":
            raise ValueError(f"Draft {draft_id} is already finalized")

        approval_jd = float(Time.now().jd)
        ra_c = float(draft.get("CENTEROFFIELDRA") or 0.0)
        de_c = float(draft.get("CENTEROFFIELDDE") or 0.0)
        obs_id = self.generate_hashtag(
            id_equipments=int(draft["ID_EQUIPMENTS"]),
            id_telescope=int(draft["ID_TELESCOPE"]),
            id_location=int(draft["ID_LOCATION"]),
            id_scanning=int(draft["ID_SCANNING"]),
            center_of_field_ra=ra_c,
            center_of_field_de=de_c,
            observation_start_jd=float(draft.get("OBSERVATIONSTARTJD") or 0.0),
        )

        arch = draft.get("ARCHIVE_PATH")
        arch_s = str(arch).strip() if arch is not None else ""
        is_cal = 1

        self.conn.execute(
            """
            INSERT INTO OBSERVATION (
                ID,
                DRAFT_ID,
                ID_EQUIPMENTS,
                ID_TELESCOPE,
                ID_LOCATION,
                ID_SCANNING,
                CENTEROFFIELDRA,
                CENTEROFFIELDDE,
                OBSERVATIONSTARTJD,
                ARCHIVE_PATH,
                IS_CALIBRATED,
                APPROVED_BY,
                APPROVAL_NOTES,
                APPROVAL_JD,
                STATUS
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'FINALIZED')
            ON CONFLICT(DRAFT_ID) DO UPDATE SET
                STATUS = 'FINALIZED',
                APPROVED_BY = excluded.APPROVED_BY,
                APPROVAL_NOTES = excluded.APPROVAL_NOTES,
                APPROVAL_JD = excluded.APPROVAL_JD
            """,
            (
                obs_id,
                int(draft_id),
                int(draft["ID_EQUIPMENTS"]),
                int(draft["ID_TELESCOPE"]),
                int(draft["ID_LOCATION"]),
                int(draft["ID_SCANNING"]),
                ra_c,
                de_c,
                float(draft.get("OBSERVATIONSTARTJD") or 0.0),
                arch_s or None,
                is_cal,
                approved_by,
                notes,
                approval_jd,
            ),
        )
        out = self.conn.execute(
            "SELECT ID FROM OBSERVATION WHERE DRAFT_ID = ? LIMIT 1;",
            (int(draft_id),),
        ).fetchone()
        if out is None:
            raise RuntimeError(f"finalize_draft_to_observation: missing OBSERVATION row for draft {draft_id}")
        final_id = str(out["ID"] if isinstance(out, sqlite3.Row) else out[0])

        self.conn.execute(
            """
            UPDATE OBS_FILES
               SET OBSERVATION_ID = ?, DRAFT_ID = NULL
             WHERE DRAFT_ID = ?;
            """,
            (final_id, int(draft_id)),
        )
        self.conn.execute(
            """
            UPDATE OBS_DRAFT
               SET STATUS = 'FINALIZED',
                   FINAL_OBSERVATION_ID = ?
             WHERE ID = ?;
            """,
            (final_id, int(draft_id)),
        )
        self.conn.commit()
        return final_id

    def _migrate_comp_library_tables(self) -> None:
        """Create FIELD_REGISTRY / COMP_STAR_LIBRARY and indexes (idempotent)."""
        if getattr(self, "_comp_library_tables_ready", False):
            return
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS FIELD_REGISTRY (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                CENTER_RA_DEG REAL NOT NULL,
                CENTER_DE_DEG REAL NOT NULL,
                RADIUS_DEG REAL NOT NULL DEFAULT 1.0,
                OBJECT_NAME TEXT,
                MASTERSTAR_PATH TEXT,
                GRID_PATH TEXT,
                COMPARISON_CSV_PATH TEXT,
                VARIABLE_TARGETS_CSV_PATH TEXT,
                FIRST_OBSERVATION_ID TEXT,
                LAST_OBSERVATION_ID TEXT,
                N_OBSERVATIONS INTEGER DEFAULT 1,
                CREATED_JD REAL,
                LAST_UPDATED_JD REAL
            );

            CREATE TABLE IF NOT EXISTS COMP_STAR_LIBRARY (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                FIELD_ID INTEGER REFERENCES FIELD_REGISTRY(ID),
                CATALOG_ID TEXT NOT NULL,
                NAME TEXT,
                RA_DEG REAL,
                DEC_DEG REAL,
                G_MAG REAL,
                BP_RP REAL,
                APERTURE_MEDIAN_MAG REAL,
                APERTURE_RMS REAL,
                PSF_MEDIAN_MAG REAL,
                PSF_RMS REAL,
                N_OBSERVATIONS INTEGER DEFAULT 1,
                N_FRAMES_TOTAL INTEGER DEFAULT 0,
                VSX_KNOWN_VARIABLE INTEGER DEFAULT 0,
                CATALOG_KNOWN_VARIABLE INTEGER DEFAULT 0,
                IS_SAFE_COMP INTEGER DEFAULT 1,
                VERDICT TEXT DEFAULT 'Approved',
                FIRST_USED_JD REAL,
                LAST_USED_JD REAL,
                LAST_OBSERVATION_ID TEXT,
                NOTES TEXT
            );

            CREATE INDEX IF NOT EXISTS IX_COMP_STAR_LIBRARY_CATALOG_ID
                ON COMP_STAR_LIBRARY(CATALOG_ID);

            CREATE INDEX IF NOT EXISTS IX_COMP_STAR_LIBRARY_FIELD_ID
                ON COMP_STAR_LIBRARY(FIELD_ID);

            CREATE INDEX IF NOT EXISTS IX_FIELD_REGISTRY_COORDS
                ON FIELD_REGISTRY(CENTER_RA_DEG, CENTER_DE_DEG);
            """
        )
        self.conn.commit()
        self._comp_library_tables_ready = True

    @staticmethod
    def _haversine_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        ra1, dec1, ra2, dec2 = map(math.radians, [ra1, dec1, ra2, dec2])
        dra = ra2 - ra1
        ddec = dec2 - dec1
        a = math.sin(ddec / 2) ** 2 + math.cos(dec1) * math.cos(dec2) * math.sin(dra / 2) ** 2
        return math.degrees(2 * math.asin(min(1.0, math.sqrt(max(0.0, a)))))

    def find_matching_field(
        self,
        ra_deg: float,
        dec_deg: float,
        *,
        match_radius_deg: float = 0.5,
    ) -> dict[str, Any] | None:
        self._migrate_comp_library_tables()
        cur = self.conn.execute("SELECT * FROM FIELD_REGISTRY;")
        rows = cur.fetchall()
        if not rows:
            return None
        best: dict[str, Any] | None = None
        best_sep = float("inf")
        for r in rows:
            d = dict(r)
            try:
                cra = float(d["CENTER_RA_DEG"])
                cde = float(d["CENTER_DE_DEG"])
            except (KeyError, TypeError, ValueError):
                continue
            sep = self._haversine_deg(float(ra_deg), float(dec_deg), cra, cde)
            if sep < best_sep:
                best_sep = sep
                best = d
        if best is not None and best_sep < float(match_radius_deg):
            return best
        return None

    def register_or_update_field(
        self,
        *,
        ra_deg: float,
        dec_deg: float,
        object_name: str | None = None,
        masterstar_path: str | None = None,
        grid_path: str | None = None,
        comparison_csv_path: str | None = None,
        variable_targets_csv_path: str | None = None,
        observation_id: str | None = None,
        radius_deg: float = 1.0,
        match_radius_deg: float = 0.5,
    ) -> int:
        from astropy.time import Time

        now_jd = float(Time.now().jd)
        match = self.find_matching_field(float(ra_deg), float(dec_deg), match_radius_deg=match_radius_deg)
        if match is not None:
            fid = int(match["ID"])
            sets = [
                "N_OBSERVATIONS = N_OBSERVATIONS + 1",
                "LAST_OBSERVATION_ID = ?",
                "LAST_UPDATED_JD = ?",
            ]
            params: list[Any] = [observation_id, now_jd]
            if object_name is not None:
                sets.append("OBJECT_NAME = ?")
                params.append(object_name)
            if masterstar_path is not None:
                sets.append("MASTERSTAR_PATH = ?")
                params.append(masterstar_path)
            if grid_path is not None:
                sets.append("GRID_PATH = ?")
                params.append(grid_path)
            if comparison_csv_path is not None:
                sets.append("COMPARISON_CSV_PATH = ?")
                params.append(comparison_csv_path)
            if variable_targets_csv_path is not None:
                sets.append("VARIABLE_TARGETS_CSV_PATH = ?")
                params.append(variable_targets_csv_path)
            params.append(fid)
            self.conn.execute(
                f"UPDATE FIELD_REGISTRY SET {', '.join(sets)} WHERE ID = ?;",
                tuple(params),
            )
            self.conn.commit()
            return fid

        self.conn.execute(
            """
            INSERT INTO FIELD_REGISTRY (
                CENTER_RA_DEG, CENTER_DE_DEG, RADIUS_DEG, OBJECT_NAME, MASTERSTAR_PATH, GRID_PATH,
                COMPARISON_CSV_PATH, VARIABLE_TARGETS_CSV_PATH,
                FIRST_OBSERVATION_ID, LAST_OBSERVATION_ID, N_OBSERVATIONS,
                CREATED_JD, LAST_UPDATED_JD
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """,
            (
                float(ra_deg),
                float(dec_deg),
                float(radius_deg),
                object_name,
                masterstar_path,
                grid_path,
                comparison_csv_path,
                variable_targets_csv_path,
                observation_id,
                observation_id,
                now_jd,
                now_jd,
            ),
        )
        self.conn.commit()
        return int(self.conn.execute("SELECT last_insert_rowid();").fetchone()[0])

    def upsert_comp_star_library(
        self,
        field_id: int,
        comp_stars: list[dict[str, Any]],
        *,
        observation_id: str | None = None,
    ) -> int:
        from astropy.time import Time

        self._migrate_comp_library_tables()
        now_jd = float(Time.now().jd)
        n_done = 0
        for star in comp_stars:
            cid = str(star.get("catalog_id") or "").strip()
            if not cid:
                continue
            verdict = str(star.get("verdict") or "Approved")
            is_safe = 1 if verdict == "Approved" else 0
            vsx = 1 if bool(star.get("vsx_known_variable")) else 0
            ckv = 1 if bool(star.get("catalog_known_variable")) else 0
            n_frames = int(star.get("n_frames") or 0)

            row = self.conn.execute(
                """
                SELECT ID, N_OBSERVATIONS, N_FRAMES_TOTAL, APERTURE_RMS, PSF_RMS,
                       APERTURE_MEDIAN_MAG, PSF_MEDIAN_MAG,
                       VSX_KNOWN_VARIABLE, CATALOG_KNOWN_VARIABLE
                FROM COMP_STAR_LIBRARY
                WHERE FIELD_ID = ? AND CATALOG_ID = ?;
                """,
                (int(field_id), cid),
            ).fetchone()
            if row is None:
                self.conn.execute(
                    """
                    INSERT INTO COMP_STAR_LIBRARY (
                        FIELD_ID, CATALOG_ID, NAME, RA_DEG, DEC_DEG, G_MAG, BP_RP,
                        APERTURE_MEDIAN_MAG, APERTURE_RMS, PSF_MEDIAN_MAG, PSF_RMS,
                        N_OBSERVATIONS, N_FRAMES_TOTAL,
                        VSX_KNOWN_VARIABLE, CATALOG_KNOWN_VARIABLE, IS_SAFE_COMP, VERDICT,
                        FIRST_USED_JD, LAST_USED_JD, LAST_OBSERVATION_ID
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        int(field_id),
                        cid,
                        star.get("name"),
                        star.get("ra_deg"),
                        star.get("dec_deg"),
                        star.get("g_mag"),
                        star.get("bp_rp"),
                        star.get("aperture_median_mag"),
                        star.get("aperture_rms"),
                        star.get("psf_median_mag"),
                        star.get("psf_rms"),
                        n_frames,
                        vsx,
                        ckv,
                        is_safe,
                        verdict,
                        now_jd,
                        now_jd,
                        observation_id,
                    ),
                )
                n_done += 1
                continue

            sid = int(row["ID"])
            old_n_obs = int(row["N_OBSERVATIONS"] or 1)
            old_n_fr = int(row["N_FRAMES_TOTAL"] or 0)
            old_ar = row["APERTURE_RMS"]
            old_pr = row["PSF_RMS"]
            old_am = row["APERTURE_MEDIAN_MAG"]
            old_pm = row["PSF_MEDIAN_MAG"]
            vsx_u = max(int(row["VSX_KNOWN_VARIABLE"] or 0), vsx)
            ckv_u = max(int(row["CATALOG_KNOWN_VARIABLE"] or 0), ckv)

            new_ar = star.get("aperture_rms")
            new_pr = star.get("psf_rms")
            w_old = max(1, old_n_fr if old_n_fr > 0 else old_n_obs)
            w_new = max(1, n_frames if n_frames > 0 else 1)

            def _pool_rms(old_v: Any, new_v: Any, wo: int, wn: int) -> float | None:
                try:
                    if new_v is None or (isinstance(new_v, float) and not math.isfinite(new_v)):
                        return float(old_v) if old_v is not None and math.isfinite(float(old_v)) else None
                    nv = float(new_v)
                    if old_v is None or (isinstance(old_v, float) and not math.isfinite(float(old_v))):
                        return nv
                    ov = float(old_v)
                    wt = wo + wn
                    if wt <= 0:
                        return nv
                    return math.sqrt(max(0.0, (wo * ov * ov + wn * nv * nv) / wt))
                except (TypeError, ValueError):
                    return None

            comb_ar = _pool_rms(old_ar, new_ar, w_old, w_new)
            comb_pr = _pool_rms(old_pr, new_pr, w_old, w_new)

            new_am = star.get("aperture_median_mag")
            comb_am = old_am
            if new_am is not None:
                try:
                    nv = float(new_am)
                    if old_am is None:
                        comb_am = nv
                    else:
                        comb_am = (float(old_am) * old_n_obs + nv) / float(old_n_obs + 1)
                except (TypeError, ValueError):
                    comb_am = old_am

            new_pm = star.get("psf_median_mag")
            comb_pm = old_pm
            if new_pm is not None:
                try:
                    nv = float(new_pm)
                    if old_pm is None:
                        comb_pm = nv
                    else:
                        comb_pm = (float(old_pm) * old_n_obs + nv) / float(old_n_obs + 1)
                except (TypeError, ValueError):
                    comb_pm = old_pm

            self.conn.execute(
                """
                UPDATE COMP_STAR_LIBRARY SET
                    NAME = COALESCE(?, NAME),
                    RA_DEG = COALESCE(?, RA_DEG),
                    DEC_DEG = COALESCE(?, DEC_DEG),
                    G_MAG = COALESCE(?, G_MAG),
                    BP_RP = COALESCE(?, BP_RP),
                    APERTURE_MEDIAN_MAG = ?,
                    APERTURE_RMS = ?,
                    PSF_MEDIAN_MAG = ?,
                    PSF_RMS = ?,
                    N_OBSERVATIONS = N_OBSERVATIONS + 1,
                    N_FRAMES_TOTAL = N_FRAMES_TOTAL + ?,
                    VSX_KNOWN_VARIABLE = ?,
                    CATALOG_KNOWN_VARIABLE = ?,
                    IS_SAFE_COMP = ?,
                    VERDICT = ?,
                    LAST_USED_JD = ?,
                    LAST_OBSERVATION_ID = ?
                WHERE ID = ?;
                """,
                (
                    star.get("name"),
                    star.get("ra_deg"),
                    star.get("dec_deg"),
                    star.get("g_mag"),
                    star.get("bp_rp"),
                    comb_am,
                    comb_ar,
                    comb_pm,
                    comb_pr,
                    n_frames,
                    vsx_u,
                    ckv_u,
                    is_safe,
                    verdict,
                    now_jd,
                    observation_id,
                    sid,
                ),
            )
            n_done += 1
        self.conn.commit()
        return n_done

    def get_comp_stars_for_field(
        self,
        ra_deg: float,
        dec_deg: float,
        *,
        match_radius_deg: float = 0.5,
        only_approved: bool = True,
    ) -> list[dict[str, Any]]:
        fld = self.find_matching_field(float(ra_deg), float(dec_deg), match_radius_deg=match_radius_deg)
        if fld is None:
            return []
        fid = int(fld["ID"])
        flt = 1 if only_approved else 0
        cur = self.conn.execute(
            """
            SELECT * FROM COMP_STAR_LIBRARY
            WHERE FIELD_ID = ?
              AND (? = 0 OR IS_SAFE_COMP = 1)
              AND (? = 0 OR VERDICT = 'Approved')
            ORDER BY G_MAG ASC;
            """,
            (fid, flt, flt),
        )
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()

