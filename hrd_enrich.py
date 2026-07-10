"""Online Gaia DR3 / SIMBAD enrichment for HRD extreme-object candidates only."""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event

logger = logging.getLogger(__name__)

_GAIA_BATCH = 50
# DR4 migration: replace gaiadr3.gaia_source TAP table name and cache filename when DR4 ships.


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        if hasattr(val, "mask") and bool(getattr(val, "mask", False)):
            return None
        x = float(val)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.is_file():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.debug("HRD enrich cache read failed (%s): %s", cache_path, exc)
    return {}


def _save_cache(cache_path: Path, data: dict[str, dict[str, Any]]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        logger.debug("HRD enrich cache write failed (%s): %s", cache_path, exc)


def _normalize_ids(source_ids: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in source_ids:
        sid = normalize_gaia_source_id(raw)
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _fetch_gaia_tap(source_ids: list[str], *, timeout_s: float) -> dict[str, dict[str, Any]]:
    if not source_ids:
        return {}

    def _run() -> dict[str, dict[str, Any]]:
        from astroquery.gaia import Gaia  # noqa: PLC0415

        Gaia.ROW_LIMIT = max(len(source_ids), _GAIA_BATCH)
        out: dict[str, dict[str, Any]] = {}
        for i0 in range(0, len(source_ids), _GAIA_BATCH):
            chunk = source_ids[i0 : i0 + _GAIA_BATCH]
            id_list = ",".join(str(int(s)) for s in chunk)
            adql = (
                "SELECT source_id, teff_gspphot, logg_gspphot "
                f"FROM gaiadr3.gaia_source WHERE source_id IN ({id_list})"
            )
            job = Gaia.launch_job_async(adql)
            table = job.get_results()
            if table is None or len(table) == 0:
                continue
            for row in table:
                sid = normalize_gaia_source_id(row["source_id"])
                if not sid:
                    continue
                teff = row["teff_gspphot"]
                logg = row["logg_gspphot"]
                out[sid] = {
                    "teff_gspphot": _safe_float(teff),
                    "logg_gspphot": _safe_float(logg),
                    "fetched_at_utc": _utc_now_iso(),
                    "enrich_source": "gaia_tap",
                }
        return out

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run)
        try:
            return fut.result(timeout=max(1.0, float(timeout_s)))
        except FuturesTimeoutError as exc:
            raise TimeoutError(f"Gaia TAP timed out after {timeout_s}s") from exc


def _fetch_simbad(source_ids: list[str], *, timeout_s: float) -> dict[str, dict[str, Any]]:
    if not source_ids:
        return {}

    def _run() -> dict[str, dict[str, Any]]:
        from astroquery.simbad import Simbad  # noqa: PLC0415

        s = Simbad()
        try:
            s.add_votable_fields("otype", "main_id")
        except Exception as exc:  # noqa: BLE001 - SIMBAD server field list varies by version
            logger.debug("SIMBAD votable fields (otype/main_id) unavailable: %s", exc)
            s.add_votable_fields("otype")
        out: dict[str, dict[str, Any]] = {}
        for sid in source_ids:
            ident = f"Gaia DR3 {sid}"
            try:
                res = s.query_object(ident)
            except Exception as exc:  # noqa: BLE001 - per-star fail-open
                logger.debug("SIMBAD query failed for %s: %s", ident, exc)
                out[sid] = {
                    "simbad_main_id": None,
                    "simbad_otype": None,
                    "fetched_at_utc": _utc_now_iso(),
                    "enrich_source": "n/a",
                }
                continue
            if res is None or len(res) == 0:
                out[sid] = {
                    "simbad_main_id": None,
                    "simbad_otype": None,
                    "fetched_at_utc": _utc_now_iso(),
                    "enrich_source": "n/a",
                }
                continue
            row = res[0]
            main_id = str(row.get("MAIN_ID") or row.get("main_id") or "").strip() or None
            otype = str(row.get("OTYPE") or row.get("otype") or "").strip() or None
            out[sid] = {
                "simbad_main_id": main_id,
                "simbad_otype": otype,
                "fetched_at_utc": _utc_now_iso(),
                "enrich_source": "simbad",
            }
        return out

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run)
        try:
            return fut.result(timeout=max(1.0, float(timeout_s)))
        except FuturesTimeoutError as exc:
            raise TimeoutError(f"SIMBAD timed out after {timeout_s}s") from exc


def enrich_candidates(
    candidates_df: pd.DataFrame,
    cache_path: Path | None,
    *,
    enabled: bool = True,
    simbad_enabled: bool = True,
    timeout_s: float = 20.0,
) -> pd.DataFrame:
    """Merge online teff/logg (Gaia TAP) and SIMBAD otype into candidate rows (fail-open)."""
    if candidates_df is None or candidates_df.empty:
        return candidates_df if candidates_df is not None else pd.DataFrame()

    out = candidates_df.copy()
    if "catalog_id" not in out.columns:
        out["catalog_id"] = ""
    out["catalog_id"] = out["catalog_id"].map(normalize_gaia_source_id)

    if not enabled:
        out["enrich_source"] = "local"
        return out

    ids = _normalize_ids(out["catalog_id"].tolist())
    cache_file = Path(cache_path) if cache_path is not None else None
    cache: dict[str, dict[str, Any]] = _load_cache(cache_file) if cache_file is not None else {}

    need_gaia = [sid for sid in ids if sid not in cache or "teff_gspphot" not in cache.get(sid, {})]
    need_simbad = (
        [sid for sid in ids if sid not in cache or "simbad_otype" not in cache.get(sid, {})]
        if simbad_enabled
        else []
    )

    skip_reason: str | None = None
    if need_gaia:
        try:
            gaia_hits = _fetch_gaia_tap(need_gaia, timeout_s=timeout_s)
            for sid, payload in gaia_hits.items():
                prev = cache.get(sid, {})
                prev.update(payload)
                cache[sid] = prev
            for sid in need_gaia:
                if sid not in gaia_hits:
                    prev = cache.get(sid, {})
                    prev.setdefault("teff_gspphot", None)
                    prev.setdefault("logg_gspphot", None)
                    prev["fetched_at_utc"] = _utc_now_iso()
                    prev["enrich_source"] = prev.get("enrich_source") or "n/a"
                    cache[sid] = prev
        except Exception as exc:  # noqa: BLE001 - network/TAP fail-open for whole enrich pass
            skip_reason = f"Gaia TAP: {exc}"
            logger.debug("HRD Gaia enrich failed: %s", exc)

    if simbad_enabled and need_simbad and skip_reason is None:
        try:
            sim_hits = _fetch_simbad(need_simbad, timeout_s=timeout_s)
            for sid, payload in sim_hits.items():
                prev = cache.get(sid, {})
                prev.update(payload)
                cache[sid] = prev
            for sid in need_simbad:
                if sid not in sim_hits:
                    prev = cache.get(sid, {})
                    prev.setdefault("simbad_main_id", None)
                    prev.setdefault("simbad_otype", None)
                    prev["fetched_at_utc"] = _utc_now_iso()
                    cache[sid] = prev
        except Exception as exc:  # noqa: BLE001 - SIMBAD fail-open; Gaia rows still usable
            if skip_reason is None:
                skip_reason = f"SIMBAD: {exc}"
            logger.debug("HRD SIMBAD enrich failed: %s", exc)

    if skip_reason:
        log_event(f"HRD enrichment skipped: {skip_reason}")

    if cache_file is not None and cache:
        _save_cache(cache_file, cache)

    teff_vals: list[Any] = []
    logg_vals: list[Any] = []
    simbad_ids: list[str] = []
    simbad_otypes: list[str] = []
    src_vals: list[str] = []

    for _, row in out.iterrows():
        sid = normalize_gaia_source_id(row.get("catalog_id", ""))
        hit = cache.get(sid, {})
        teff_local = row.get("teff_gspphot")
        logg_local = row.get("logg_gspphot")
        teff = hit.get("teff_gspphot") if hit.get("teff_gspphot") is not None else teff_local
        logg = hit.get("logg_gspphot") if hit.get("logg_gspphot") is not None else logg_local
        src = str(hit.get("enrich_source") or "local")
        if skip_reason and src == "local" and teff is None and logg is None:
            src = "n/a"
        teff_vals.append(teff)
        logg_vals.append(logg)
        simbad_ids.append(str(hit.get("simbad_main_id") or ""))
        simbad_otypes.append(str(hit.get("simbad_otype") or ""))
        src_vals.append(src)

    out["teff_gspphot"] = teff_vals
    out["logg_gspphot"] = logg_vals
    out["simbad_main_id"] = simbad_ids
    out["simbad_otype"] = simbad_otypes
    out["enrich_source"] = src_vals
    return out
