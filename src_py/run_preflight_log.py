# -*- coding: ascii -*-
"""Persist RUN VYVAR pre-infolog failures under ``<data_dir>/logs/``."""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PREFLIGHT_LOG_PREFIX = "run_preflight_error_"


def preflight_logs_dir(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs


def summarize_db_fk_state(db: Any, cfg: Any | None = None) -> str:
    """Short FK-relevant DB snapshot for preflight error logs."""
    lines: list[str] = []
    if db is None:
        return "database: (none)"
    conn = getattr(db, "conn", None)
    if conn is None:
        return "database: (no connection)"
    for table in ("EQUIPMENTS", "TELESCOPE", "LOCATION", "SCANNING"):
        try:
            n = int(conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0])
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{table}: (query failed: {exc})")
            continue
        if table in ("EQUIPMENTS", "TELESCOPE", "LOCATION") and n > 0:
            ids = [
                int(r[0])
                for r in conn.execute(f"SELECT ID FROM {table} ORDER BY ID;").fetchall()
            ]
            lines.append(f"{table}: {n} row(s), ids={ids}")
        else:
            lines.append(f"{table}: {n} row(s)")
    if cfg is not None:
        try:
            loc_id = int(getattr(cfg, "observer_location_id", 0) or 0)
        except (TypeError, ValueError):
            loc_id = 0
        lines.append(f"config observer_location_id={loc_id}")
        if loc_id > 0 and hasattr(db, "_fk_row_exists"):
            try:
                exists = bool(db._fk_row_exists("LOCATION", loc_id))
            except Exception as exc:  # noqa: BLE001
                exists = f"? ({exc})"
            lines.append(f"observer_location_id={loc_id} exists_in_db={exists}")
    return "\n".join(lines)


def write_run_preflight_error_log(
    data_root: str | Path,
    *,
    step: str,
    exc: BaseException,
    db: Any | None = None,
    cfg: Any | None = None,
    failing_statement: str | None = None,
) -> Path:
    """Write ``logs/run_preflight_error_<timestamp>.log`` and return its path."""
    logs = preflight_logs_dir(data_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = logs / f"{_PREFLIGHT_LOG_PREFIX}{ts}.log"
    stmt = (failing_statement or "").strip() or _infer_failing_statement(exc)
    body = "\n".join(
        [
            f"# VYVAR RUN VYVAR preflight error - {ts} UTC",
            f"step: {step}",
            f"exception: {type(exc).__name__}: {exc}",
            "",
            "failing_statement:",
            stmt,
            "",
            "database_state:",
            summarize_db_fk_state(db, cfg),
            "",
            "traceback:",
            traceback.format_exc(),
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")
    return path


def _infer_failing_statement(exc: BaseException) -> str:
    msg = str(exc)
    if "INSERT INTO OBS_DRAFT" in msg:
        return msg
    if "FOREIGN KEY constraint failed" in msg:
        return (
            "sqlite INSERT with FOREIGN KEY constraint failed "
            "(likely OBS_DRAFT id_location/id_scanning or stale config observer_location_id)"
        )
    if "Observation references missing" in msg or "missing database row" in msg:
        return msg
    return msg or type(exc).__name__
