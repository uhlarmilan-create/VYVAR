"""Moved from photometry_core.py (CONSOLIDATE-01E3). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import AbstractSet, Any
import json
import logging
import math
import os
from astropy.io import fits as astrofits
import numpy as np
import pandas as pd
from comp_pool_rms import attach_comp_rms_to_pool_rows, compute_global_pool_rms_map
from comp_rms_loo import (
    COMP_RMS_FRAMES_BASIS,
    COMP_RMS_LOO_PHOTON_K_DEFAULT,
    LN10_OVER_2P5,
    compute_loo_mag_rms_map,
)
from config import AppConfig
from database import query_local_gaia_by_source_ids
from gaia_catalog_id import normalize_gaia_source_id, read_vyvar_csv
from infolog import log_event
from photometry_provenance import merge_photometry_pipeline_meta
from photometry_shared import _normalize_gaia_id, _safe_polyfit, _target_display_name
from photometry_core import (
    LOGGER,
    _GAIA_ID_DTYPE,
    _PHASE_USECOLS_PERFRAME,
)

LAST_EXCLUDED_TARGETS: pd.DataFrame = pd.DataFrame(
    columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"]
)

def _sid_int(v: Any) -> int | None:
    sid = normalize_gaia_source_id(v)
    if sid and sid.isdigit():
        try:
            return int(sid)
        except Exception:  # noqa: BLE001
            # EXC-0121: T4 -- Non-integer Gaia source_id returns None - downstream treats star as uncatalogued (EXCEPT-BULK 2026-07-08)
            return None
    return None

def _enrich_comp_bp_rp(
    candidates: pd.DataFrame,
    gaia_db_path: str | None,
    *,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Doplni ``bp_rp`` pre comp hviezdy kde chyba (Gaia DR3 podla ``source_id``)."""
    if candidates is None or getattr(candidates, "empty", True):
        return pd.DataFrame()

    df = candidates.copy()
    if "bp_rp" not in df.columns:
        df["bp_rp"] = float("nan")

    df["bp_rp"] = pd.to_numeric(df.get("bp_rp"), errors="coerce")
    if "ra_deg" in df.columns:
        df["ra_deg"] = pd.to_numeric(df.get("ra_deg"), errors="coerce")
    if "dec_deg" in df.columns:
        df["dec_deg"] = pd.to_numeric(df.get("dec_deg"), errors="coerce")

    con = None
    gaia_cols: set[str] = set()
    gaia_path = str(gaia_db_path or "").strip()
    if gaia_path and os.path.exists(gaia_path):
        try:
            import sqlite3  # noqa: PLC0415

            con = sqlite3.connect(gaia_path)
            con.row_factory = sqlite3.Row
            gaia_cols = {
                str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()
            }
        except Exception:  # noqa: BLE001
            con = None
            gaia_cols = set()

    sel_bp = "bp_rp" in gaia_cols
    gaia_bp_cache: dict[int, float] = {}

    def _gaia_bp_rp(sid_i: int) -> float:
        if sid_i in gaia_bp_cache:
            return gaia_bp_cache[sid_i]
        bp_r = float("nan")
        gid_pf = normalize_gaia_source_id(sid_i)
        if gaia_prefetch and gid_pf and gid_pf in gaia_prefetch:
            try:
                vbp = gaia_prefetch[gid_pf].get("bp_rp")
                if vbp is not None and math.isfinite(float(vbp)):
                    bp_r = float(vbp)
            except (TypeError, ValueError):
                pass
            gaia_bp_cache[int(sid_i)] = bp_r
            return bp_r
        if con is not None and sel_bp:
            try:
                rw = con.execute(
                    "SELECT bp_rp FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                    (int(sid_i),),
                ).fetchone()
                if rw is not None and rw["bp_rp"] is not None:
                    bp_r = float(rw["bp_rp"])
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0122] Gaia DB bp_rp row fetch fails - comp star keeps NaN bp_rp and wrong colour tier: %s', exc)
                pass
        gaia_bp_cache[int(sid_i)] = bp_r if math.isfinite(bp_r) else float("nan")
        return gaia_bp_cache[int(sid_i)]

    try:
        for idx, row in df.iterrows():
            try:
                bp_now = float(pd.to_numeric(row.get("bp_rp"), errors="coerce"))
            except Exception:  # noqa: BLE001
                bp_now = float("nan")
            if math.isfinite(bp_now):
                continue
            sid_i = _sid_int(row.get("source_id") or row.get("catalog_id") or row.get("name"))
            if sid_i is None:
                continue
            gaia_bp = _gaia_bp_rp(sid_i)
            if math.isfinite(gaia_bp):
                df.at[idx, "bp_rp"] = float(gaia_bp)
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:  # noqa: BLE001
            # EXC-0123: T2 -- sqlite con.close() failure during comp bp_rp enrichment ignored (EXCEPT-BULK-2 2026-07-08)
            pass

    return df

def _ensure_active_target_display_names(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    if "vsx_name" not in out.columns:
        out["vsx_name"] = ""
    filled: list[str] = []
    for _, row in out.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", ""))
        filled.append(_target_display_name(row, fallback_cid=cid))
    out["vsx_name"] = filled
    if "name" in out.columns:
        blank_name = out["name"].astype(str).str.strip().str.lower().isin(("", "nan", "none"))
        out.loc[blank_name, "name"] = out.loc[blank_name, "catalog_id"]
    return out

def _variable_targets_looks_like_ct_presel_stub(vt_path: Path, *, masterstars_csv: Path) -> bool:
    if not vt_path.is_file() or not masterstars_csv.is_file():
        return False
    try:
        vt = pd.read_csv(vt_path, low_memory=False, nrows=500)
        ms = pd.read_csv(masterstars_csv, low_memory=False, usecols=["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0138] variable_targets stub detection CSV read fails - returns False (stub may go undetected): %s', exc)
        return False
    if len(vt) >= max(80, int(len(ms) * 0.05)):
        return False
    if "notes" in vt.columns:
        notes = vt["notes"].astype(str).str.contains("CT presel", case=False, na=False)
        if bool(notes.any()):
            return True
    if "name" in vt.columns:
        names = vt["name"].astype(str)
        if bool(names.str.contains("M67 in-range|M67 red-giant", case=False, regex=True).any()):
            return True
    return len(vt) < 50 and len(ms) > 200

def ensure_full_variable_targets_if_presel_stub(
    *,
    variable_targets_csv: Path,
    masterstars_csv: Path,
    masterstar_fits: Path,
    cfg: Any | None = None,
    draft_id: int | None = None,
) -> bool:
    """Restore full-field ``variable_targets.csv`` when CT presel stub replaced production list."""
    vt_path = Path(variable_targets_csv)
    ms_path = Path(masterstars_csv)
    if not _variable_targets_looks_like_ct_presel_stub(vt_path, masterstars_csv=ms_path):
        return False
    ps_dir = vt_path.parent
    ms_fits = Path(masterstar_fits)
    if not ms_fits.is_file():
        ms_fits = ps_dir / "MASTERSTAR.fits"
    if not ms_fits.is_file() or not ms_path.is_file():
        logging.warning(
            "[PHOT] CT presel stub detected but cannot restore variable_targets (missing MASTERSTAR/masterstars)"
        )
        return False
    try:
        from pipeline import write_photometry_plan_files  # noqa: PLC0415

        _cfg = cfg or AppConfig()
        write_photometry_plan_files(
            platesolve_dir=ps_dir,
            masterstar_fits=ms_fits,
            masterstars_csv=ms_path,
            n_comparison_stars=int(getattr(_cfg, "comparison_stars_pool_n", 0) or 0),
            require_non_variable=bool(getattr(_cfg, "phase01_comparison_require_non_variable", True)),
            draft_id=int(draft_id) if draft_id is not None else None,
            database_path=getattr(_cfg, "database_path", None),
        )
        log_event(
            f"[PHOT] Restored full variable_targets.csv from field cone (replaced CT presel stub in {ps_dir.name})"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0139] Full variable_targets restore from field cone fails - CT presel stub may remain as targ...: %s', exc)
        logging.warning("[PHOT] variable_targets restore failed: %s", exc)
        return False

def _normalize_id_value(x: Any) -> str:
    """Normalize Gaia-like IDs loaded as floats; keep non-numeric strings."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:  # noqa: BLE001
        return s

def _normalize_id_series(s: pd.Series) -> pd.Series:
    return s.apply(_normalize_id_value)

def _bool_col(series: pd.Series) -> pd.Series:
    """Normalizuje stlpec na bool bez ohladu na True/False/'true'/'false'/1/0."""
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))

def _phase0_effective_frame_hw_px(
    vt: pd.DataFrame,
    ms: pd.DataFrame,
    *,
    frame_w_px: int,
    frame_h_px: int,
    edge_margin_px: int,
) -> tuple[int, int]:
    """``frame_w_px`` / ``frame_h_px`` z volania alebo vacsie - podla max. x,y v VT a masterstars.

    Predvolene 2082x1397 casto nezodpovedaju velkemu cipu; inak sa VSX ciele s velkymi pixelmi
    (napr. DY Peg) vylucia este pred cross-matchom, **bez** ohladu na ``vsx_type`` (ziadny filter na SXPHE).
    """
    xs: list[float] = []
    ys: list[float] = []
    for df in (vt, ms):
        if "x" in df.columns and "y" in df.columns:
            xs.extend(pd.to_numeric(df["x"], errors="coerce").dropna().astype(float).tolist())
            ys.extend(pd.to_numeric(df["y"], errors="coerce").dropna().astype(float).tolist())
    if not xs or not ys:
        return int(frame_w_px), int(frame_h_px)
    em = int(edge_margin_px)
    need_w = int(math.ceil(float(max(xs)))) + em + 2
    need_h = int(math.ceil(float(max(ys)))) + em + 2
    return max(int(frame_w_px), need_w), max(int(frame_h_px), need_h)

def _active_target_zone_flag(ms_row: pd.Series, zone_val_raw: str) -> str:
    """Mapovanie masterstars ``zone`` (+ legacy ``is_saturated``) na ``zone_flag`` pre active_targets."""
    z = str(zone_val_raw or "").strip().lower()
    if z in ("linear", "noise", "saturated"):
        return z
    if z in ("noisy1", "noisy2", "noisy3"):
        return "noise"
    try:
        sat = bool(ms_row.get("is_saturated", False))
    except Exception:  # noqa: BLE001
        # EXC-0191: T3 -- Nested log_event inside catalog_id auto-repair failure also fails - repair error messag... (EXCEPT-BULK-2 2026-07-08)
        sat = False
    if sat:
        return "saturated"
    if not z:
        return "neznama_zona"
    return z

def _auto_repair_catalog_ids(
    *,
    vt_path: Path,
    gaia_db_path: str | None,
    log_fn: Any = None,
    max_sep_arcsec: float = 10.0,
) -> dict[str, Any]:
    """Auto-repair poskodene Gaia catalog_id v variable_targets.csv podla RA/DEC.

    Bezpecnostne pravidla:
    - Ak `gaia_db_path` nie je nastavena alebo DB neexistuje -> nic nerob.
    - Ak `variable_targets.csv` nema `catalog_id` alebo RA/DEC -> nic nerob.
    - Opravuj iba vtedy, ked najblizsi Gaia zdroj je dostatocne blizko (`max_sep_arcsec`).
    - Vytvor `.bak` zalohu iba ak sa nieco realne opravilo.
    """
    try:
        from repair_catalog_ids import repair_catalog_ids_from_gaia_db  # noqa: PLC0415

        _log = log_fn or log_event
        if not gaia_db_path:
            return {"ok": False, "reason": "no_gaia_db_path"}
        dbp = Path(str(gaia_db_path))
        if not dbp.is_file():
            return {"ok": False, "reason": "gaia_db_missing", "gaia_db_path": str(dbp)}
        if not Path(vt_path).is_file():
            return {"ok": False, "reason": "vt_missing", "vt_path": str(vt_path)}
        res = repair_catalog_ids_from_gaia_db(
            variable_targets_csv=Path(vt_path),
            gaia_db_path=dbp,
            backup=True,
            max_sep_arcsec=float(max_sep_arcsec),
            log_fn=_log,
        )
        if int(res.get("repaired") or 0) > 0:
            _log(f"[COMP] auto-repair variable_targets.csv: repaired={res.get('repaired')} warnings={res.get('warnings')}")
        return res
    except Exception as exc:  # noqa: BLE001
        try:
            (log_fn or log_event)(f"[COMP] auto-repair variable_targets.csv FAILED: {exc!s}")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "reason": "exception", "error": str(exc)}

def _enrich_active_targets_bp_rp(
    targets_df: pd.DataFrame,
    *,
    gaia_db_path: str | Path | None,
) -> pd.DataFrame:
    """Dopln ``bp_rp`` pre active targets z Gaia DR3 podla ``catalog_id``."""
    if targets_df is None or getattr(targets_df, "empty", True):
        return targets_df

    df = targets_df.copy()
    if "bp_rp" not in df.columns:
        df["bp_rp"] = float("nan")
    df["bp_rp"] = pd.to_numeric(df["bp_rp"], errors="coerce")
    if "ra_deg" in df.columns:
        df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
    if "dec_deg" in df.columns:
        df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")

    gaia_path = str(gaia_db_path or "").strip()
    con = None
    gaia_cols: set[str] = set()
    if gaia_path and os.path.exists(gaia_path):
        try:
            import sqlite3  # noqa: PLC0415

            con = sqlite3.connect(gaia_path)
            con.row_factory = sqlite3.Row
            gaia_cols = {
                str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()
            }
        except Exception:  # noqa: BLE001
            # EXC-0193: T2 -- sqlite con.close() after active-target bp_rp enrichment ignored (EXCEPT-BULK-2 2026-07-08)
            con = None
            gaia_cols = set()
    sel_bp = "bp_rp" in gaia_cols
    gaia_cache: dict[int, float] = {}

    def _gaia_bp(sid_i: int) -> float:
        if sid_i in gaia_cache:
            return gaia_cache[sid_i]
        bp_r = float("nan")
        if con is not None and sel_bp:
            try:
                rw = con.execute(
                    "SELECT bp_rp FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                    (int(sid_i),),
                ).fetchone()
                if rw is not None and rw["bp_rp"] is not None:
                    bp_r = float(rw["bp_rp"])
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0192] Gaia DB bp_rp fetch for active targets fails - target row keeps NaN bp_rp: %s', exc)
                pass
        gaia_cache[int(sid_i)] = bp_r if math.isfinite(bp_r) else float("nan")
        return gaia_cache[int(sid_i)]

    try:
        for idx, row in df.iterrows():
            try:
                bp_now = float(pd.to_numeric(row.get("bp_rp"), errors="coerce"))
            except Exception:  # noqa: BLE001
                bp_now = float("nan")
            if math.isfinite(bp_now):
                continue
            sid_i = _sid_int(row.get("catalog_id"))
            if sid_i is None:
                continue
            gaia_bp = _gaia_bp(sid_i)
            if math.isfinite(gaia_bp):
                df.at[idx, "bp_rp"] = float(gaia_bp)
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:  # noqa: BLE001
            # EXC-0195: T4 -- DB manifest files[] NAXIS query fails - returns caller-supplied default frame width/height (EXCEPT-BULK-2 2026-07-08)
            pass

    return df

def _resolve_frame_hw_px_from_masterstar(
    ms_fits: Path,
    *,
    frame_w_px: int,
    frame_h_px: int,
    db: Any = None,
    draft_id: int | None = None,
) -> tuple[int, int, str]:
    """Authoritative chip width/height for Phase 0+1 spatial culling.

    Priority: (1) MASTERSTAR FITS ``NAXIS1``/``NAXIS2``; (2) light FITS via draft;
    (3) caller defaults (global cfg knob / hardcoded 2082x1397).
    """
    w_def, h_def = int(frame_w_px), int(frame_h_px)
    if ms_fits.is_file():
        try:
            with astrofits.open(ms_fits, memmap=False) as hdul:
                hdr = hdul[0].header
                w = int(hdr.get("NAXIS1", 0) or 0)
                h = int(hdr.get("NAXIS2", 0) or 0)
                if w > 0 and h > 0:
                    return w, h, "fits_naxis"
        except Exception:  # noqa: BLE001
            pass
    if db is not None and draft_id is not None:
        try:
            did = int(draft_id)
        except (TypeError, ValueError):
            did = 0
        if did > 0 and hasattr(db, "fetch_draft_light_rows_for_quality"):
            try:
                light_rows = db.fetch_draft_light_rows_for_quality(did)
                fp0 = None
                for lr in light_rows:
                    fp0 = lr.get("FILE_PATH")
                    if fp0:
                        break
                if fp0:
                    from astropy.io import fits as _fits_nax

                    with _fits_nax.open(str(fp0), memmap=False) as _hdul:
                        w = int(_hdul[0].header.get("NAXIS1") or 0)
                        h = int(_hdul[0].header.get("NAXIS2") or 0)
                    if w > 0 and h > 0:
                        return w, h, "fits_naxis_light"
            except Exception:  # noqa: BLE001
                pass
    return w_def, h_def, "caller_default"

def _read_field_density_inputs(
    ms_fits: Path,
    masterstars_csv: Path,
    frame_w_px: int,
    frame_h_px: int,
) -> tuple[int, int, int, str, int | None]:
    """Vrati ``(n_stars, chip_w, chip_h, source, n_stars_dao_raw)`` pre hustotu pola.

    ``n_stars``: pocet riadkov v ``masterstars_csv`` s neprazdnym ``catalog_id`` (Gaia-matched).
    ``n_stars_dao_raw``: ``VY_NDAO`` z MASTERSTAR FITS (iba referencia / JSON, nie klasifikacia).
    ``source``: ``masterstars_gaia_matched`` | ``VY_NDAO_fallback`` | ``defaults``.
    """
    cw, ch = int(frame_w_px), int(frame_h_px)
    n_stars = 0
    src = "defaults"
    vy_ndao_raw: int | None = None
    cw, ch, _hw_src = _resolve_frame_hw_px_from_masterstar(
        ms_fits, frame_w_px=cw, frame_h_px=ch
    )
    if _hw_src == "fits_naxis":
        src = "fits_naxis"
    if ms_fits.is_file():
        try:
            with astrofits.open(ms_fits, memmap=False) as hdul:
                hdr = hdul[0].header
                v = hdr.get("VY_NDAO")
                if v is not None and str(v).strip() != "":
                    try:
                        vy_ndao_raw = int(float(v))
                    except (TypeError, ValueError):
                        vy_ndao_raw = None
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0196] VY_NDAO header read from MASTERSTAR fails - field density uses masterstars count fallback: %s', exc)
            pass
    msc_path = Path(masterstars_csv)
    if msc_path.is_file():
        try:
            _msc_df = pd.read_csv(
                msc_path,
                usecols=["catalog_id"],
                low_memory=False,
                dtype={"catalog_id": str},
            )
            _cid = _msc_df["catalog_id"].astype(str).str.strip()
            _n_gaia = int((~_cid.isin(["", "nan", "None"])).sum())
            if _n_gaia > 0:
                n_stars = _n_gaia
                src = "masterstars_gaia_matched"
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0197] masterstars_full_match.csv star-count read fails - field density n_stars may use VY_NDA...: %s', exc)
            pass
    if n_stars <= 0 and vy_ndao_raw is not None and vy_ndao_raw > 0:
        n_stars = int(vy_ndao_raw)
        src = "VY_NDAO_fallback"
    return int(max(0, n_stars)), cw, ch, src, vy_ndao_raw

def _refresh_variable_targets_xy(
    variable_targets_csv: Path,
    wcs: Any,
    chip_w: int,
    chip_h: int,
) -> None:
    """Prepocita x/y stlpce variable_targets.csv z aktualneho MASTERSTAR WCS."""
    from astropy.wcs import WCS

    if wcs is None or not isinstance(wcs, WCS):
        return
    vt_path = Path(variable_targets_csv)
    if not vt_path.is_file():
        return

    logging.debug("[VT REFRESH] frame %sx%s px (MASTERSTAR -> VT x,y)", chip_w, chip_h)

    df = pd.read_csv(vt_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
    if "ra_deg" in df.columns:
        ra = pd.to_numeric(df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    elif "ra" in df.columns:
        ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        logging.warning("[VT REFRESH] chybaju stlpce ra_deg / ra - x/y neaktualizovane")
        return
    if "dec_deg" in df.columns:
        dec = pd.to_numeric(df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    elif "dec" in df.columns:
        dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        logging.warning("[VT REFRESH] chybaju stlpce dec_deg / dec - x/y neaktualizovane")
        return

    try:
        ok = np.isfinite(ra) & np.isfinite(dec)
        xy = np.full((len(df), 2), np.nan, dtype=np.float64)
        if bool(ok.any()):
            pts = np.column_stack([ra[ok], dec[ok]])
            xy[ok, :] = wcs.all_world2pix(pts, 0)
        df["x"] = xy[:, 0]
        df["y"] = xy[:, 1]
        df.to_csv(vt_path, index=False)
    except (ValueError, TypeError, AttributeError) as e:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().vt_wcs_refresh_fail += 1
        logging.error("[VT REFRESH] WCS prepocet zlyhal: %s - x/y ostavaju stale", e)
        return

    logging.info("[VT REFRESH] x/y suradnice variable_targets.csv aktualizovane z MASTERSTAR WCS")
    xv = df["x"].to_numpy(dtype=np.float64, copy=False)
    yv = df["y"].to_numpy(dtype=np.float64, copy=False)
    if np.isfinite(xv).any() and np.isfinite(yv).any():
        logging.info(
            "[VT REFRESH] %d riadkov, x=[%.0f,%.0f] y=[%.0f,%.0f]",
            len(df),
            float(np.nanmin(xv)),
            float(np.nanmax(xv)),
            float(np.nanmin(yv)),
            float(np.nanmax(yv)),
        )
    else:
        logging.info("[VT REFRESH] %d riadkov (ziadne platne x/y po prepocte)", len(df))

def _attach_predicted_dilution_report(active: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Report-only Gaia-predicted dilution fraction (design D; does not gate or correct)."""
    out = active.copy()
    out["predicted_dilution_factor"] = 1.0
    gdb = str(cfg.gaia_db_path or "").strip()
    if out.empty or not gdb:
        return out
    try:
        from dilution import compute_dilution_factor  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return out
    ap = float(cfg.gs11_dilution_aperture_arcsec)
    mld = float(cfg.gs11_dilution_mag_limit_delta)
    factors: list[float] = []
    for _, row in out.iterrows():
        try:
            res = compute_dilution_factor(
                float(row["ra_deg"]),
                float(row["dec_deg"]),
                float(pd.to_numeric(row.get("mag"), errors="coerce")),
                ap,
                gdb,
                catalog_id=row.get("catalog_id"),
                mag_limit_delta=mld,
            )
            factors.append(float(res.get("dilution_factor", 1.0)))
        except Exception:  # noqa: BLE001
            factors.append(1.0)
    out["predicted_dilution_factor"] = factors
    return out

def select_active_targets(
    variable_targets_csv: Path,
    masterstars_csv: Path,
    *,
    frame_w_px: int = 2082,
    frame_h_px: int = 1397,
    edge_margin_px: int = 50,
    safe_bbox: tuple[float, float, float, float] | None = None,
    gaia_db_path: str | None = None,
    vsx_local_db_path: str | None = None,
    masterstar_fits_path: Path | str | None = None,
    plate_scale_arcsec_px: float | None = None,
    cfg: Any | None = None,
    target_depth_g: float | None = None,
) -> pd.DataFrame:
    """Faza 0: Filtruj VSX premenne -> active_targets.

    Pravidla:
    - Hviezda musi byt v snimke (``x,y`` aspon ``edge_margin_px`` od okraja efektivneho pola; to iste cislo
      ako ``chip_interior_margin_px`` vo Faze 0+1 - jednotne s porovnavackami a suspected).
    - Sirka/vyska sa zvacsi z dat ak treba
    - VSX auto rows: identity join on planner ``catalog_id`` in masterstars (no positional matching).
      Only ``gaia_match_source=masterstars`` rows are promotable; ``gaia_dr3_direct`` / ``no_match`` stay
      in variable_targets for comp veto but are excluded here.
    - Manual / exoplanet targets: identity join on ``catalog_id`` when present (not subject to VSX gate).
    - ``catalog_id`` z masterstars musi byt neprazdny (inak sa ciel vynecha).
    - ``catalog_only`` / ``neznama_zona`` / ``saturated`` / ``noise`` zone flags mask photometry
      (TARGET-DEPTH-02: ``noise`` = below DAO N-sigma on MASTERSTAR; flag, do not omit).
    - ``vsx_out_of_scope_types`` (config): VSX auto-selected targets whose type tokens match
      are kept in active_targets with ``skip_photometry=True`` and
      ``skip_reason='vsx_type_out_of_scope'`` (mask-first). Manual targets are never filtered.
      Empty list = inactive (byte-identical to prior behaviour).
    - ``target_depth_g`` (optional): MASTERSTAR-derived population depth; fainter targets get
      ``skip_reason='below_target_depth'``.
    Returns:
        DataFrame s active targets - stlpce z variable_targets + pridane zo masterstars:
        [name, catalog_id, ra_deg, dec_deg, vsx_name, vsx_type, vsx_period,
         x, y, mag, b_v, bp_rp, zone_flag, skip_photometry, skip_reason]
    """
    global LAST_EXCLUDED_TARGETS
    # Auto-repair poskodenych Gaia ID pred nacitanim (ak je dostupna lokalna Gaia DB).
    _auto_repair_catalog_ids(vt_path=Path(variable_targets_csv), gaia_db_path=gaia_db_path, log_fn=log_event)

    # variable_targets.csv moze prist zvonka a casto ma catalog_id ako float/scientific - citaj ako string
    vt = pd.read_csv(variable_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    # masterstars_full_match.csv casto nesie presny Gaia source_id v "name" aj ked catalog_id je poskodeny floatom
    ms = pd.read_csv(masterstars_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    if "catalog_id" in vt.columns:
        vt["catalog_id"] = vt["catalog_id"].apply(_normalize_gaia_id)
    # Normalizuj Gaia ID na string.
    # POZOR: "name" v masterstars casto obsahuje presny Gaia source_id; nesmieme ho prehnat cez float().
    if "catalog_id" in ms.columns:
        ms["catalog_id"] = _normalize_id_series(ms["catalog_id"])
    if "name" in ms.columns:
        ms["name"] = ms["name"].fillna("").astype(str).str.strip()

    # Normalizuj bool stlpce v masterstars
    for col in ("is_usable", "is_saturated", "is_noisy", "snr50_ok", "likely_saturated"):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    fw, fh = _phase0_effective_frame_hw_px(
        vt, ms, frame_w_px=int(frame_w_px), frame_h_px=int(frame_h_px), edge_margin_px=int(edge_margin_px)
    )
    if fw != int(frame_w_px) or fh != int(frame_h_px):
        logging.info(
            "[FAZA 0] Rozmer cipu zvacseny z %sx%s na %sx%s px (max x,y z variable_targets/masterstars + okraj)",
            int(frame_w_px),
            int(frame_h_px),
            fw,
            fh,
        )

    # Filter: v snimke (annulus-aware safe bbox, else fixed edge margin)
    vt["x"] = pd.to_numeric(vt["x"], errors="coerce")
    vt["y"] = pd.to_numeric(vt["y"], errors="coerce")
    if safe_bbox is not None:
        try:
            from aperture_policy import stars_fit_on_chip  # noqa: PLC0415

            x0b, y0b, x1b, y1b = safe_bbox
            before = int(len(vt))
            in_frame = stars_fit_on_chip(
                vt["x"], vt["y"], (0.0, 0.0, 0.0), (float(x0b), float(y0b), float(x1b), float(y1b))
            )
            in_frame = pd.Series(np.asarray(in_frame, dtype=bool), index=vt.index)
            removed = before - int(in_frame.sum())
            if removed > 0:
                logging.info(
                    f"[BORDER] Active targets: removed {removed} rows outside safe bbox "
                    f"(annulus-aware intersection)"
                )
        except Exception:  # noqa: BLE001
            in_frame = (
                vt["x"].between(edge_margin_px, fw - edge_margin_px)
                & vt["y"].between(edge_margin_px, fh - edge_margin_px)
            )
    else:
        in_frame = (
            vt["x"].between(edge_margin_px, fw - edge_margin_px)
            & vt["y"].between(edge_margin_px, fh - edge_margin_px)
        )
    vt_in = vt[in_frame].copy()

    _cfg = cfg if cfg is not None else AppConfig()

    from vsx_type_scope import (  # noqa: PLC0415
        is_vsx_auto_selected_target,
        vsx_type_is_out_of_scope,
    )

    # Identity join index: normalized Gaia source_id -> masterstar row.
    ms_by_cid: dict[str, pd.Series] = {}
    for _, ms_row in ms.iterrows():
        cid_ms = normalize_gaia_source_id(ms_row.get("name", ""))
        if not cid_ms:
            cid_ms = _normalize_gaia_id(ms_row.get("catalog_id", ""))
        if cid_ms and cid_ms not in ms_by_cid:
            ms_by_cid[str(cid_ms)] = ms_row

    out_of_frame = int(len(vt) - int(in_frame.sum()))
    no_catalog_id = 0
    no_gaia_id = 0
    no_dao_detection = 0
    not_target_eligible = 0
    matched_rows: list[dict] = []
    matched_vt_idx: set[Any] = set()
    excluded_rows: list[dict[str, Any]] = []

    def _excluded_target_row(vrow: pd.Series, reason: str, *, mag: float | None = None) -> dict[str, Any]:
        mag_val = mag
        if mag_val is None:
            mag_val = float(pd.to_numeric(vrow.get("mag", float("nan")), errors="coerce"))
        return {
            "name": str(vrow.get("name", "") or ""),
            "vsx_name": str(vrow.get("vsx_name", "") or ""),
            "vsx_type": str(vrow.get("vsx_type", "") or ""),
            "ra_deg": float(vrow.get("ra_deg", float("nan"))),
            "dec_deg": float(vrow.get("dec_deg", float("nan"))),
            "mag": mag_val,
            "reason": reason,
        }

    for _, vrow_off in vt.loc[~in_frame].iterrows():
        excluded_rows.append(_excluded_target_row(vrow_off, "out_of_frame"))

    for vidx, vrow in vt_in.iterrows():
        ra_v = float(pd.to_numeric(vrow.get("ra_deg"), errors="coerce"))
        dec_v = float(pd.to_numeric(vrow.get("dec_deg"), errors="coerce"))
        if not (math.isfinite(ra_v) and math.isfinite(dec_v)):
            continue

        is_vsx_auto = is_vsx_auto_selected_target(vrow)
        planner_cid = _normalize_gaia_id(vrow.get("catalog_id", ""))
        gaia_src = str(vrow.get("gaia_match_source", "") or "").strip().lower()

        if is_vsx_auto:
            if gaia_src not in ("masterstars",):
                not_target_eligible += 1
                excluded_rows.append(
                    _excluded_target_row(vrow, "no_gaia_id" if not planner_cid else "not_target_eligible")
                )
                continue
            if not planner_cid:
                no_gaia_id += 1
                excluded_rows.append(_excluded_target_row(vrow, "no_gaia_id"))
                continue
        elif not planner_cid:
            no_catalog_id += 1
            excluded_rows.append(_excluded_target_row(vrow, "no_catalog_id"))
            continue

        ms_row = ms_by_cid.get(str(planner_cid))
        if ms_row is None:
            no_dao_detection += 1
            excluded_rows.append(_excluded_target_row(vrow, "no_dao_detection"))
            continue

        catalog_id_norm = str(planner_cid)
        zone_val_raw = str(ms_row.get("zone", "")).strip()
        zone_flag = _active_target_zone_flag(ms_row, zone_val_raw)
        mag_for_skip = float(
            pd.to_numeric(
                ms_row.get("mag", ms_row.get("phot_g_mean_mag", float("nan"))),
                errors="coerce",
            )
        )
        _snr_raw = ms_row.get("snr50_ok", True)
        if _snr_raw is None or (isinstance(_snr_raw, float) and not math.isfinite(float(_snr_raw))):
            snr50_ok_for_skip = True
        else:
            snr50_ok_for_skip = bool(_bool_col(pd.Series([_snr_raw])).iloc[0])
        if (not snr50_ok_for_skip) and math.isfinite(mag_for_skip) and mag_for_skip < 8.0:
            logging.info(
                "[SKIP] %s: mag=%.1f snr50_ok=False - pravdepodobne saturovana, skip",
                catalog_id_norm,
                mag_for_skip,
            )
            excluded_rows.append(_excluded_target_row(vrow, "saturated", mag=mag_for_skip))
            continue
        skip_ph = zone_flag in ("saturated", "catalog_only", "neznama_zona", "noise")
        if zone_flag == "noise":
            skip_reason = "zone_noise"
        else:
            skip_reason = "zone_flag" if skip_ph else ""
        # TARGET-DEPTH-02: draft-derived MASTERSTAR population depth (flag, do not omit).
        if (
            (not skip_ph)
            and target_depth_g is not None
            and math.isfinite(float(target_depth_g))
            and math.isfinite(mag_for_skip)
            and float(mag_for_skip) > float(target_depth_g)
        ):
            skip_ph = True
            skip_reason = "below_target_depth"
        _voos = list(getattr(_cfg, "vsx_out_of_scope_types", []) or [])
        if (
            (not skip_ph)
            and _voos
            and is_vsx_auto
            and vsx_type_is_out_of_scope(str(vrow.get("vsx_type", "") or ""), _voos)
        ):
            skip_ph = True
            skip_reason = "vsx_type_out_of_scope"
        rec = {
            "name": vrow.get("name", ""),
            "vsx_name": vrow.get("vsx_name", ""),
            "vsx_type": vrow.get("vsx_type", ""),
            "vsx_period": vrow.get("vsx_period", ""),
            "priority": vrow.get("priority", 1),
            "ra_deg": ra_v,
            "dec_deg": dec_v,
            "x": float(pd.to_numeric(ms_row.get("x", vrow.get("x")), errors="coerce")),
            "y": float(pd.to_numeric(ms_row.get("y", vrow.get("y")), errors="coerce")),
            "catalog_id": catalog_id_norm,
            "mag": mag_for_skip,
            "b_v": float(ms_row.get("b_v", float("nan"))),
            "bp_rp": float(ms_row.get("bp_rp", float("nan"))),
            "zone_flag": zone_flag,
            "skip_photometry": bool(skip_ph),
            "skip_reason": str(skip_reason),
        }
        if "catalog" in vrow.index:
            rec["catalog"] = vrow.get("catalog", "")
        for _exo_col in (
            "exo_host_obj_id",
            "exo_host_name",
            "exo_cat_source",
            "exo_disposition",
            "exo_match_sep_arcsec",
            "target_origin",
        ):
            if _exo_col in vrow.index:
                rec[_exo_col] = vrow.get(_exo_col, "")
        matched_vt_idx.add(vidx)
        matched_rows.append(rec)

    _empty_cols = [
        "name",
        "vsx_name",
        "vsx_type",
        "vsx_period",
        "priority",
        "ra_deg",
        "dec_deg",
        "x",
        "y",
        "catalog_id",
        "mag",
        "b_v",
        "bp_rp",
        "zone_flag",
        "skip_photometry",
        "skip_reason",
    ]
    n_masked_zone = int(sum(1 for r in matched_rows if r.get("skip_reason") == "zone_flag"))
    n_masked_noise = int(sum(1 for r in matched_rows if r.get("skip_reason") == "zone_noise"))
    n_masked_vsx_type = int(sum(1 for r in matched_rows if r.get("skip_reason") == "vsx_type_out_of_scope"))
    n_masked_depth = int(sum(1 for r in matched_rows if r.get("skip_reason") == "below_target_depth"))
    n_gaia_id_assigned = int(vt_in["catalog_id"].apply(lambda x: bool(_normalize_gaia_id(x))).sum()) if "catalog_id" in vt_in.columns else 0
    _contam_pct = float("nan")
    if not matched_rows:
        LAST_EXCLUDED_TARGETS = (
            pd.DataFrame(excluded_rows)
            if excluded_rows
            else pd.DataFrame(columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"])
        )
        log_event(
            "select_active_targets: linear=0 noise=0 saturated=0 "
            f"no_catalog_id={no_catalog_id} no_gaia_id={no_gaia_id} no_dao_detection={no_dao_detection} "
            f"out_of_frame={out_of_frame}"
        )
        logging.info(
            "FAZA 0 funnel: vsx_bbox=%d -> in_frame=%d -> gaia_id_assigned=%d (contamination=%s) "
            "-> dao_detected=0 -> active=0 | excluded: no_dao_detection=%d no_gaia_id=%d "
            "not_target_eligible=%d out_of_frame=%d | masked: zone_flag=0 vsx_type_out_of_scope=0",
            int(len(vt)),
            int(len(vt_in)),
            n_gaia_id_assigned,
            "n/a",
            no_dao_detection,
            no_gaia_id,
            not_target_eligible,
            out_of_frame,
        )
        return pd.DataFrame(columns=_empty_cols)

    result = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame(columns=_empty_cols)
    if "catalog_id" in result.columns:
        # NEPOUZIVAT float() (precision loss). Pouzi robustnu normalizaciu.
        result["catalog_id"] = result["catalog_id"].apply(_normalize_gaia_id)
    # Gaia BP-RP + rovnaka B-V hierarchia ako pre comp (nesmie prepisat vsetky b_v z NaN bp_rp v masterstars).
    result = _enrich_active_targets_bp_rp(
        result,
        gaia_db_path=gaia_db_path,
    )
    # Deduplicate by catalog_id - keep row with real VSX name over
    # Gaia-placeholder (e.g. "V0842 Her" preferred over
    # "Gaia DR3 1400549806859236864")
    if "catalog_id" in result.columns:
        # Prefer rows where vsx_name / name does NOT start with "Gaia DR3"
        _is_gaia_placeholder = (
            result.get("vsx_name", result.get("name", pd.Series(dtype=str)))
            .astype(str)
            .str.startswith("Gaia DR3")
        )
        # Sort: non-Gaia-placeholder first, then keep first per catalog_id
        result = (
            result
            .assign(_gaia_placeholder=_is_gaia_placeholder.astype(int))
            .sort_values("_gaia_placeholder")
            .drop_duplicates(subset=["catalog_id"], keep="first")
            .drop(columns=["_gaia_placeholder"])
            .reset_index(drop=True)
        )
        log_event(
            f"select_active_targets: deduped to {len(result)} unique "
            f"catalog_ids (prefer real VSX name over Gaia placeholder)"
        )
    n_lin = int((result["zone_flag"] == "linear").sum())
    n_noise = int((result["zone_flag"] == "noise").sum())
    n_sat = int((result["zone_flag"] == "saturated").sum())
    log_event(
        f"select_active_targets: linear={n_lin} noise={n_noise} "
        f"saturated={n_sat} no_catalog_id={no_catalog_id} no_gaia_id={no_gaia_id} "
        f"no_dao_detection={no_dao_detection} out_of_frame={out_of_frame}"
    )
    _funnel_msg = (
        f"FAZA 0 funnel: vsx_bbox={int(len(vt))} -> in_frame={int(len(vt_in))} -> "
        f"gaia_id_assigned={n_gaia_id_assigned} (contamination="
        f"{f'{_contam_pct:.1f}%' if math.isfinite(_contam_pct) else 'n/a'}) -> "
        f"dao_detected={len(matched_rows)} -> active={len(result)} | excluded: "
        f"no_dao_detection={no_dao_detection} no_gaia_id={no_gaia_id} "
        f"not_target_eligible={not_target_eligible} out_of_frame={out_of_frame} | masked: "
        f"zone_flag={n_masked_zone} zone_noise={n_masked_noise} "
        f"vsx_type_out_of_scope={n_masked_vsx_type} below_target_depth={n_masked_depth}"
    )
    logging.info(_funnel_msg)
    log_event(_funnel_msg)
    _voos_cfg = list(getattr(_cfg, "vsx_out_of_scope_types", []) or [])
    if _voos_cfg and n_masked_vsx_type == 0 and "catalog" in vt.columns:
        _vt_vsx = vt[vt["catalog"].astype(str).str.upper() == "VSX"]
        _types = set()
        for _t in _vt_vsx.get("vsx_type", pd.Series(dtype=str)).astype(str):
            for _tok in _t.replace(":", " ").replace("/", " ").split():
                _types.add(_tok.strip().upper())
        if any(str(t).strip().upper() in _types for t in _voos_cfg):
            logging.warning(
                "[FAZA 0] vsx_out_of_scope_types=%s but zero rows masked - check VSX type tokens",
                _voos_cfg,
            )
    if n_gaia_id_assigned > 0 and len(vt_in) > 0:
        _assign_frac = float(n_gaia_id_assigned) / float(len(vt_in))
        if _assign_frac < 0.50:
            logging.warning(
                "[FAZA 0] gaia_id_assigned=%d far below in_frame=%d (%.0f%%) - sparse Gaia cross-match",
                n_gaia_id_assigned,
                int(len(vt_in)),
                100.0 * _assign_frac,
            )
    LAST_EXCLUDED_TARGETS = (
        pd.DataFrame(excluded_rows)
        if excluded_rows
        else pd.DataFrame(columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"])
    )
    result = _ensure_active_target_display_names(result)
    return result.reset_index(drop=True)

def _batch_enrich_targets_bp_rp_from_gaia_db(
    target_cids: list[str],
    gaia_db_path: str,
) -> dict[str, dict[str, Any]]:
    """Prefetch Gaia ``bp_rp`` / ``teff_gspphot`` for Phase 1 targets (batched SQL)."""
    gdb = str(gaia_db_path or "").strip()
    if not target_cids or not gdb:
        return {}
    try:
        gp = Path(gdb).expanduser().resolve()
        if not gp.is_file():
            return {}
    except OSError:
        return {}

    ids_norm: list[str] = []
    seen: set[str] = set()
    for raw in target_cids:
        g = normalize_gaia_source_id(raw)
        if not g or not g.isdigit() or g in seen:
            continue
        seen.add(g)
        ids_norm.append(g)
    if not ids_norm:
        return {}

    out: dict[str, dict[str, Any]] = {}
    try:
        base = query_local_gaia_by_source_ids(gp, ids_norm)
        for k, v in base.items():
            out[k] = {
                "bp_rp": v.get("bp_rp"),
                "teff_gspphot": None,
            }
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 1] Batch Gaia bp_rp lookup failed: %s", exc)
        return {}

    try:
        import sqlite3  # noqa: PLC0415

        con = sqlite3.connect(str(gp))
        con.row_factory = sqlite3.Row
        try:
            cols = {
                str(r[1]).strip().lower()
                for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()
            }
            if "teff_gspphot" not in cols:
                return out
            ids_int = [int(x) for x in ids_norm]
            bs = 500
            for i0 in range(0, len(ids_int), bs):
                chunk = ids_int[i0 : i0 + bs]
                ph = ",".join("?" * len(chunk))
                q = f"SELECT source_id, teff_gspphot FROM gaia_dr3 WHERE source_id IN ({ph});"
                for row in con.execute(q, chunk):
                    key = normalize_gaia_source_id(row["source_id"])
                    if not key or key not in out:
                        continue
                    te = row["teff_gspphot"]
                    if te is not None:
                        try:
                            out[key]["teff_gspphot"] = float(te)
                        except (TypeError, ValueError):
                            pass
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 1] Batch Gaia teff lookup failed: %s", exc)

    return out

def _enrich_target_bp_rp_from_gaia_db(
    target: pd.Series,
    *,
    gaia_db_path: str,
    vsx_local_db_path: str | None = None,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
) -> pd.Series:
    """Dopln ``bp_rp`` pre jeden active target (Faza 1) z Gaia podla ``source_id``."""
    out = target.copy()
    vsx = str(out.get("vsx_name", "") or "").strip() or str(out.get("name", "") or "").strip() or "?"

    def _fscalar(key: str) -> float:
        try:
            v = float(pd.to_numeric(out.get(key), errors="coerce"))
        except Exception:  # noqa: BLE001
            return float("nan")
        return v if math.isfinite(v) else float("nan")

    bpr_ms = _fscalar("bp_rp")

    gid = normalize_gaia_source_id(out.get("catalog_id"))
    gdb = str(gaia_db_path or "").strip()
    try:
        gp = Path(gdb).expanduser().resolve()
        gdb_ok = bool(gdb) and gp.is_file()
    except OSError:
        gdb_ok = False

    bpr_nf = float("nan")
    _prefetched = bool(gaia_prefetch and gid and gid in gaia_prefetch)
    if _prefetched:
        pf = gaia_prefetch[gid]  # type: ignore[index]
        try:
            vbp = pf.get("bp_rp")
            if vbp is not None and math.isfinite(float(vbp)):
                bpr_nf = float(vbp)
        except (TypeError, ValueError):
            pass
    elif gid and gdb_ok and gid.isdigit():
        try:
            import sqlite3  # noqa: PLC0415

            con = sqlite3.connect(str(gp))
            con.row_factory = sqlite3.Row
            try:
                cols = {str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()}
                parts = [c for c in ("bp_rp", "teff_gspphot") if c in cols]
                if parts:
                    rw = con.execute(
                        f"SELECT {', '.join(parts)} FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                        (int(gid),),
                    ).fetchone()
                    if rw is not None:
                        if "bp_rp" in parts:
                            try:
                                vbp = rw["bp_rp"]
                                if vbp is not None:
                                    bpr_nf = float(vbp)
                            except (TypeError, ValueError, KeyError, IndexError):
                                pass
            finally:
                con.close()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0202] Outer Gaia SQL for target bp_rp logs failure - target keeps CSV bp_rp or NaN: %s', exc)
            log_event(f"TARGET Gaia SQL: {vsx} - {exc!s}")

    if math.isfinite(bpr_nf):
        out["bp_rp"] = float(bpr_nf)
    elif math.isfinite(bpr_ms):
        out["bp_rp"] = float(bpr_ms)
    elif not gid:
        log_event(f"TARGET bp_rp: {vsx} - bez platneho Gaia catalog_id")
    return out

def _bprp_tier_ladder_for_selection(
    cfg: AppConfig | None,
    max_delta_bprp: float,
) -> list[float]:
    """Tier-ladder colour windows: tier1 -> tier2 -> tier3 -> comp_max_delta_bprp cap."""
    if cfg is not None:
        _tier_lims = cfg.comp_tier_bprp_limits()
        raw = [
            float(_tier_lims[0]),
            float(_tier_lims[1]),
            float(_tier_lims[2]),
            float(getattr(cfg, "comp_max_delta_bprp", max_delta_bprp)),
        ]
    else:
        raw = [float(max_delta_bprp)]
    out: list[float] = []
    for v in raw:
        if math.isfinite(v) and v > 0 and v not in out:
            out.append(float(v))
    return out or [float(max_delta_bprp)]

def _select_comps_by_rms_then_color(
    candidates: pd.DataFrame,
    target_bprp: float,
    n_comp_min: int,
    n_comp_max: int,
    max_delta_bprp: float = 0.5,
    *,
    cfg: AppConfig | None = None,
    max_comp_rms: float | None = None,
    fwhm_px: float | None = None,
) -> pd.DataFrame:
    """COMP-ASSIGN-03: RMS -> colour ladder -> distance; single-source only.

    ``comp_rms`` from the global pool (step 1). ``phase01_comparison_max_comp_rms``
    is a hard ceiling before the colour ladder and ``head(n_comp_max)``
    (COMP-ASSIGN-02). Colour window widens until ``n_comp_min`` under-ceiling
    single-source comps exist; within a colour step, order is ``comp_rms``, then
    ``|delta BP-RP|``, then ``_dist_deg``, then ``catalog_id``. Blends (nearest
    catalogue neighbour closer than ``snr_cog_isolation_fwhm`` x FWHM) are
    excluded from candidacy. ``n_comp_max`` remains a ceiling, not a pad target.
    """
    if candidates is None or getattr(candidates, "empty", True):
        return pd.DataFrame()

    if "comp_rms" not in candidates.columns:
        raise ValueError("_select_comps_by_rms_then_color requires comp_rms column")

    _n_min = max(1, int(n_comp_min))
    _n_max = max(_n_min, int(n_comp_max))
    _floor = 1e-6
    _ceil = float("nan")
    _iso_fwhm = 3.0
    if cfg is not None:
        try:
            _floor = float(getattr(cfg, "comp_select_rms_floor", 1e-6) or 1e-6)
        except (TypeError, ValueError):
            _floor = 1e-6
        try:
            _ceil = float(
                getattr(cfg, "phase01_comparison_max_comp_rms", float("nan")) or float("nan")
            )
        except (TypeError, ValueError):
            _ceil = float("nan")
        try:
            _iso_fwhm = float(getattr(cfg, "snr_cog_isolation_fwhm", 3.0) or 3.0)
        except (TypeError, ValueError):
            _iso_fwhm = 3.0
    if max_comp_rms is not None:
        try:
            _mc = float(max_comp_rms)
        except (TypeError, ValueError):
            _mc = float("nan")
        if math.isfinite(_mc) and _mc > 0:
            _ceil = _mc
    if not (math.isfinite(_iso_fwhm) and _iso_fwhm > 0):
        _iso_fwhm = 3.0

    out = candidates.copy()
    rms = pd.to_numeric(out["comp_rms"], errors="coerce")
    out = out[rms >= _floor].copy()
    if out.empty:
        return out

    # Authoritative per-target RMS gate (INV-COMP-RMS-01): LOO mag MAD vs
    # k * photon (snr_ap_pixscaled) and an absolute 0.1 mag cap.
    _k_loo = COMP_RMS_LOO_PHOTON_K_DEFAULT
    if cfg is not None:
        try:
            _k_loo = float(getattr(cfg, "comp_rms_loo_photon_k", _k_loo) or _k_loo)
        except (TypeError, ValueError):
            _k_loo = COMP_RMS_LOO_PHOTON_K_DEFAULT
    _abs_ceil = _ceil if math.isfinite(_ceil) and _ceil > 0 else 0.1
    if "snr_ap_pixscaled" not in out.columns:
        raise ValueError(
            "INV-COMP-RMS-01: snr_ap_pixscaled missing on comparison candidates"
        )
    _snr = pd.to_numeric(out["snr_ap_pixscaled"], errors="coerce")
    _rms_c = pd.to_numeric(out["comp_rms"], errors="coerce")
    _ph = LN10_OVER_2P5 / _snr
    _star_ceil = np.minimum(float(_abs_ceil), float(_k_loo) * _ph)
    _n0 = int(len(out))
    _pre_ceil = out.copy()
    out = out[_rms_c.notna() & _snr.gt(0) & _rms_c.le(_star_ceil)].copy()
    logging.info(
        "[COMP] loo ceiling k=%.3f abs=%.4f: %d -> %d under-ceiling candidates",
        float(_k_loo),
        float(_abs_ceil),
        _n0,
        int(len(out)),
    )
    if _n0 > int(len(out)):
        _ceil_dropped = _pre_ceil.loc[~_pre_ceil.index.isin(out.index)]
        if not _ceil_dropped.empty:
            _bc = _ceil_dropped.sort_values("comp_rms", kind="mergesort").iloc[0]
            _cidc = "catalog_id" if "catalog_id" in _ceil_dropped.columns else _ceil_dropped.columns[0]
            logging.info(
                "[COMP] best ceiling-rejected cid=%s rms=%s threshold=%.4f",
                str(_bc.get(_cidc, "")),
                str(_bc.get("comp_rms", "")),
                float(_abs_ceil),
            )
    if out.empty:
        logging.warning(
            "[COMP] no candidates under loo ceiling k=%.3f abs=%.4f after floor filter",
            float(_k_loo),
            float(_abs_ceil),
        )
        return out

    # Single-source (COMP-ASSIGN-03): exclude blends inside CoG isolation radius.
    _nn_fwhm = None
    if "_nn_dist_fwhm" in out.columns:
        _nn_fwhm = pd.to_numeric(out["_nn_dist_fwhm"], errors="coerce")
    elif "_nn_px" in out.columns and fwhm_px is not None:
        try:
            _fw = float(fwhm_px)
        except (TypeError, ValueError):
            _fw = float("nan")
        if math.isfinite(_fw) and _fw > 0:
            _nn_fwhm = pd.to_numeric(out["_nn_px"], errors="coerce") / _fw
    if _nn_fwhm is not None:
        out = out.copy()
        if "_nn_dist_fwhm" not in out.columns:
            out["_nn_dist_fwhm"] = _nn_fwhm
        _n_pre = int(len(out))
        _pre_iso = out.copy()
        _ok = _nn_fwhm.notna() & (_nn_fwhm >= float(_iso_fwhm))
        # Missing NN: keep (unknown), only drop measured blends.
        _keep = _nn_fwhm.isna() | _ok
        out = out[_keep].copy()
        logging.info(
            "[COMP] single-source isolation >=%.2f FWHM: %d -> %d candidates",
            float(_iso_fwhm),
            _n_pre,
            int(len(out)),
        )
        if _n_pre > int(len(out)):
            _iso_dropped = _pre_iso.loc[~_pre_iso.index.isin(out.index)]
            if not _iso_dropped.empty:
                _nn_d = pd.to_numeric(_iso_dropped.get("_nn_dist_fwhm"), errors="coerce")
                if _nn_d is not None and _nn_d.notna().any():
                    _iso_dropped = _iso_dropped.assign(_nn_sort=_nn_d)
                    _bi = _iso_dropped.sort_values(
                        ["comp_rms", "_nn_sort"],
                        ascending=[True, True],
                        kind="mergesort",
                    ).iloc[0]
                    _nnv = float(pd.to_numeric(_bi.get("_nn_dist_fwhm"), errors="coerce"))
                else:
                    _bi = _iso_dropped.sort_values("comp_rms", kind="mergesort").iloc[0]
                    _nnv = float("nan")
                _cid_iso = (
                    "catalog_id" if "catalog_id" in _iso_dropped.columns else _iso_dropped.columns[0]
                )
                logging.info(
                    "[COMP] best isolation-rejected cid=%s rms=%s nn_fwhm=%s threshold=%.2f FWHM",
                    str(_bi.get(_cid_iso, "")),
                    str(_bi.get("comp_rms", "")),
                    f"{_nnv:.3f}" if math.isfinite(_nnv) else "nan",
                    float(_iso_fwhm),
                )
        if out.empty:
            logging.warning(
                "[COMP] no candidates after single-source isolation (%.2f FWHM)",
                float(_iso_fwhm),
            )
            return out

    id_col = "catalog_id" if "catalog_id" in out.columns else out.columns[0]
    tb = float(target_bprp)
    if "bp_rp" in out.columns and math.isfinite(tb):
        out["_delta_bprp_abs"] = (
            pd.to_numeric(out.get("bp_rp"), errors="coerce") - tb
        ).abs()
    else:
        out["_delta_bprp_abs"] = 0.0

    if "_dist_deg" in out.columns:
        out["_dist_sort"] = pd.to_numeric(out["_dist_deg"], errors="coerce")
    elif "dist_deg" in out.columns:
        out["_dist_sort"] = pd.to_numeric(out["dist_deg"], errors="coerce")
    else:
        out["_dist_sort"] = 0.0
    out["_dist_sort"] = out["_dist_sort"].fillna(float("inf"))
    out["_rms_sort"] = pd.to_numeric(out["comp_rms"], errors="coerce").fillna(float("inf"))
    out["_delta_bprp_abs"] = pd.to_numeric(out["_delta_bprp_abs"], errors="coerce").fillna(
        float("inf")
    )

    ladder = _bprp_tier_ladder_for_selection(cfg, float(max_delta_bprp))
    selected = pd.DataFrame()
    used_lim = float("nan")
    ladder_step = 0
    for step_i, lim in enumerate(ladder, start=1):
        pool = out[out["_delta_bprp_abs"] <= float(lim)].copy()
        if len(pool) >= _n_min:
            selected = pool
            used_lim = float(lim)
            ladder_step = int(step_i)
            logging.info(
                "[COMP] color filter at delta_bprp<=%.3f (n=%d >= n_comp_min=%d) "
                "ladder_step=%d (RMS-ordered within)",
                float(lim),
                len(pool),
                _n_min,
                ladder_step,
            )
            break
        selected = pool
        used_lim = float(lim)
        ladder_step = int(step_i)
    if selected.empty:
        selected = out
        used_lim = float(ladder[-1]) if ladder else float(max_delta_bprp)
        ladder_step = int(len(ladder)) if ladder else 0
        logging.warning(
            "[COMP] color filter relaxed to full under-ceiling single-source set "
            "(no ladder step reached n_comp_min=%d; lim=%.3f; n=%d)",
            _n_min,
            used_lim,
            int(len(selected)),
        )
    elif len(selected) < _n_min and len(out) >= _n_min:
        n_ladder = int(len(selected))
        selected = out
        ladder_step = int(len(ladder)) if ladder else ladder_step
        logging.info(
            "[COMP] color filter relaxed to full under-ceiling single-source set "
            "after ladder (n_ladder=%d < n_comp_min=%d; n_full=%d)",
            n_ladder,
            _n_min,
            int(len(out)),
        )

    # COMP-ASSIGN-03: RMS first, then colour, then distance.
    n_pre_colour = int(len(out))
    n_colour_pool = int(len(selected))
    if n_pre_colour > n_colour_pool and math.isfinite(float(used_lim)):
        _col_dropped = out.loc[~out.index.isin(selected.index)]
        if not _col_dropped.empty:
            _bcol = _col_dropped.sort_values("_rms_sort", kind="mergesort").iloc[0]
            _db = float(pd.to_numeric(_bcol.get("_delta_bprp_abs"), errors="coerce"))
            logging.info(
                "[COMP] best colour-rejected cid=%s rms=%s d_bprp=%s colour_lim=%.3f",
                str(_bcol.get(id_col, "")),
                str(_bcol.get("comp_rms", "")),
                f"{_db:.3f}" if math.isfinite(_db) else "nan",
                float(used_lim),
            )
    selected = selected.sort_values(
        ["_rms_sort", "_delta_bprp_abs", "_dist_sort", id_col],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    n_clean_pool = int(len(selected))
    _ceil_s = f"{float(_ceil):.4f}" if math.isfinite(_ceil) and _ceil > 0 else "nan"
    if n_clean_pool > _n_max:
        _nxt = selected.iloc[_n_max]
        _nr = float(pd.to_numeric(_nxt.get("comp_rms"), errors="coerce"))
        _nd = float(pd.to_numeric(_nxt.get("_delta_bprp_abs"), errors="coerce"))
        _nds = float(pd.to_numeric(_nxt.get("_dist_sort"), errors="coerce"))
        logging.info(
            "[COMP] clean pool n=%d after ceiling/isolation/colour; "
            "n_comp_max=%d admitted=%d; best not-admitted (n_comp_max/distance sort) "
            "cid=%s rms=%s d_bprp=%s dist_deg=%s "
            "(thresholds: ceiling=%s isolation=%.2f FWHM colour_lim=%s)",
            n_clean_pool,
            _n_max,
            _n_max,
            str(_nxt.get(id_col, "")),
            f"{_nr:.4f}" if math.isfinite(_nr) else "nan",
            f"{_nd:.3f}" if math.isfinite(_nd) else "nan",
            f"{_nds:.5f}" if math.isfinite(_nds) else "nan",
            _ceil_s,
            float(_iso_fwhm),
            f"{float(used_lim):.3f}" if math.isfinite(float(used_lim)) else "nan",
        )
    else:
        logging.info(
            "[COMP] clean pool n=%d after ceiling/isolation/colour; "
            "n_comp_max=%d admitted=%d; nothing better in the pool "
            "(thresholds: ceiling=%s isolation=%.2f FWHM colour_lim=%s)",
            n_clean_pool,
            _n_max,
            n_clean_pool,
            _ceil_s,
            float(_iso_fwhm),
            f"{float(used_lim):.3f}" if math.isfinite(float(used_lim)) else "nan",
        )
    selected = selected.head(_n_max).copy()
    selected.attrs["color_ladder_lim"] = used_lim
    selected.attrs["color_ladder_step"] = int(ladder_step)
    selected.attrs["max_comp_rms_ceiling"] = (
        float(_ceil) if math.isfinite(_ceil) and _ceil > 0 else float("nan")
    )
    selected.attrs["single_source_isolation_fwhm"] = float(_iso_fwhm)
    return selected.drop(columns=["_rms_sort", "_dist_sort"], errors="ignore")

def _select_comps_by_color_then_rms(
    candidates: pd.DataFrame,
    target_bprp: float,
    n_comp_min: int,
    n_comp_max: int,
    max_delta_bprp: float = 0.5,
    *,
    cfg: AppConfig | None = None,
    max_comp_rms: float | None = None,
    fwhm_px: float | None = None,
) -> pd.DataFrame:
    """Deprecated alias - COMP-ASSIGN-03 renamed to ``_select_comps_by_rms_then_color``."""
    return _select_comps_by_rms_then_color(
        candidates,
        target_bprp,
        n_comp_min,
        n_comp_max,
        max_delta_bprp,
        cfg=cfg,
        max_comp_rms=max_comp_rms,
        fwhm_px=fwhm_px,
    )

def _select_comps_tiered(
    candidates: pd.DataFrame,
    n_comp_min: int,
    n_comp_max: int,
    tier_weights: dict[int, float],
) -> tuple[pd.DataFrame, str]:
    """
    Vracia (selected_df, selection_note)

    Greedy tier-based vyber:
    T1 -> T2 -> T3 -> T4 (len ak treba)
    Nikdy nemiesaj T3/T4 ak T1+T2 >= n_comp_min.
    Sort: comp_tier ASC, comp_rms ASC, catalog_id (proximity only via max_dist_deg gate).
    """
    _ = tier_weights  # reserved for future (selection is tier/rms-only; weights affect Phase 2A)
    if candidates is None or getattr(candidates, "empty", True):
        return pd.DataFrame(), "no_candidates"

    if "comp_tier" not in candidates.columns or "comp_rms" not in candidates.columns:
        return pd.DataFrame(), "missing_cols"

    selected = pd.DataFrame()
    note = "ok"

    for max_tier in [1, 2, 3, 4]:
        pool = candidates[candidates["comp_tier"] <= max_tier].copy()
        pool["comp_tier"] = pd.to_numeric(pool["comp_tier"], errors="coerce").fillna(4).astype(int)
        pool["comp_rms"] = pd.to_numeric(pool["comp_rms"], errors="coerce")

        # Zorad: tier ASC, potom comp_rms ASC, potom catalog_id (stable tiebreak)
        pool = pool.sort_values(
            ["comp_tier", "comp_rms", "catalog_id"],
            ascending=[True, True, True],
            kind="mergesort",
        )

        # Ber max n_comp_max (vzdy)
        selected = pool.head(int(n_comp_max))

        n_t1t2 = len(selected[selected["comp_tier"] <= 2])

        if len(selected) >= int(n_comp_min):
            # Mame dostatok - ale over:
            # ak mame >= n_comp_min z T1+T2, odober T3/T4 z vyberu
            if n_t1t2 >= int(n_comp_min):
                selected = (
                    selected[selected["comp_tier"] <= 2]
                    .sort_values(
                        ["comp_tier", "comp_rms", "catalog_id"],
                        ascending=[True, True, True],
                        kind="mergesort",
                    )
                    .head(int(n_comp_max))
                )
                if max_tier == 1:
                    note = "t1_only"
                elif max_tier == 2:
                    note = "t1t2"
                else:
                    note = "t1t2"  # T3/T4 boli odobrane
            else:
                # T3/T4 boli potrebne - selected uz obsahuje az n_comp_max
                if max_tier == 3:
                    note = "t3_fallback"
                else:
                    note = "t4_fallback"
            break
    else:
        note = "sparse"

    if len(selected) == 0:
        note = "sparse_no_comps"

    return selected.reset_index(drop=True), note

def build_global_comp_pool(
    masterstars_df: pd.DataFrame,
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    variable_target_catalog_ids: AbstractSet[str] | None,
    safe_bbox: tuple[float, float, float, float] | None,
    chip_fw: int,
    chip_fh: int,
    chip_interior_margin_px: int,
    max_comp_rms: float,
    cfg: AppConfig,
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float = 0.3,
    fwhm_px: float = 3.7,
    max_psf_chi2: float = 3.0,
    max_fwhm_factor: float = 1.5,
    edge_bad_frame_frac_max: float = 0.10,
    admission_artifact_dir: Path | str | None = None,
    photometry_dir_for_meta: Path | str | None = None,
) -> pd.DataFrame:
    """Zostav globalny comp pool - staticke filtre + RMS napriec framami (raz pre pole)."""
    pool = masterstars_df.copy()
    for _id_col in ("catalog_id", "name"):
        if _id_col in pool.columns:
            pool[_id_col] = _normalize_id_series(pool[_id_col])
    for col in (
        "is_usable",
        "is_saturated",
        "is_noisy",
        "snr50_ok",
        "vsx_known_variable",
        "likely_saturated",
    ):
        if col in pool.columns:
            pool[col] = _bool_col(pool[col])

    margin = int(chip_interior_margin_px)
    if "x" not in pool.columns or "y" not in pool.columns:
        logging.warning("[GLOBAL COMP POOL] chybaju x/y - prazdny pool")
        return pd.DataFrame()
    xn = pd.to_numeric(pool["x"], errors="coerce")
    yn = pd.to_numeric(pool["y"], errors="coerce")

    if safe_bbox is not None:
        # safe_bbox already shrinks by alignment intersection + sky annulus (r_out); do not inset again.
        from aperture_policy import stars_fit_on_chip  # noqa: PLC0415

        x0, y0, x1, y1 = safe_bbox
        if float(x1) > float(x0) and float(y1) > float(y0):
            _on = stars_fit_on_chip(
                xn, yn, (0.0, 0.0, 0.0), (float(x0), float(y0), float(x1), float(y1))
            )
            pool = pool.loc[np.asarray(_on, dtype=bool)].copy()
        else:
            pool = pool.iloc[0:0].copy()
    else:
        fw, fh = int(chip_fw), int(chip_fh)
        if margin > 0 and fw > 2 * margin and fh > 2 * margin:
            pool = pool.loc[
                xn.between(float(margin), float(fw - margin)) & yn.between(float(margin), float(fh - margin))
            ].copy()

    _vt_gaia_ids: frozenset[str] | None = None
    if variable_target_catalog_ids:
        from gaia_catalog_id import normalize_gaia_id_set  # noqa: PLC0415

        _vt_gaia_ids = normalize_gaia_id_set(
            variable_target_catalog_ids,
            log_label="variable_target_catalog_ids (global comp pool)",
        ) or None
    if _vt_gaia_ids:
        nid = pool.get("catalog_id", pool.get("name", pd.Series("", index=pool.index))).map(_normalize_gaia_id)
        pool = pool.loc[~nid.isin(_vt_gaia_ids)].copy()

    if "zone" in pool.columns:
        z = pool["zone"].astype(str).str.strip().str.lower()
        pool = pool.loc[~z.isin(["saturated", "nonlinear"])].copy()

    # FORCED-PHOT / COMP-ADMIT-03 review: is_noisy is not a gate; is_usable may
    # still encode zone=linear for diagnostics but noisy stars enter via scatter.
    cand_mask = (
        ~_bool_col(pool.get("is_saturated", pd.Series(False, index=pool.index)))
        & ~_bool_col(pool.get("vsx_known_variable", pd.Series(False, index=pool.index)))
        & ~_bool_col(pool.get("likely_saturated", pd.Series(False, index=pool.index)))
    )
    pool = pool.loc[cand_mask].copy()

    from d3_comparison_candidacy import apply_d3_comparison_candidacy  # noqa: PLC0415

    _solve_rms_d3: float | None = None
    if photometry_dir_for_meta is not None:
        try:
            _pm = Path(photometry_dir_for_meta)
            _meta_p = _pm if _pm.name == "pipeline_meta.json" else _pm / "pipeline_meta.json"
            if _meta_p.is_file():
                _pj = json.loads(_meta_p.read_text(encoding="utf-8"))
                _inp = _pj.get("match_sep_formula_inputs") or {}
                if isinstance(_inp, dict) and _inp.get("solve_rms_px") is not None:
                    _solve_rms_d3 = float(_inp["solve_rms_px"])
        except Exception:  # noqa: BLE001
            _solve_rms_d3 = None
    _d3_mask, _d3_meta = apply_d3_comparison_candidacy(
        pool,
        fwhm_dao_px=float(fwhm_px),
        solve_rms_px=_solve_rms_d3,
        log_label="global_comp_pool",
    )
    pool = pool.loc[_d3_mask].copy()
    _ = _d3_meta

    # Gaia NSS = known variable; QSO/GAL = measurability (extended).
    if bool(cfg.phase01_comparison_exclude_gaia_nss) and "gaia_nss" in pool.columns:
        pool = pool.loc[~_bool_col(pool["gaia_nss"])].copy()
    if bool(cfg.phase01_comparison_exclude_gaia_extobj):
        for _ext_col in ("gaia_qso", "gaia_gal"):
            if _ext_col in pool.columns:
                pool = pool.loc[~_bool_col(pool[_ext_col])].copy()

    if pool.empty:
        logging.warning("[GLOBAL COMP POOL] po statickych filtroch 0 riadkov")
        return pool.reset_index(drop=True)

    id_col = "name" if "name" in pool.columns else "catalog_id"
    cand_ids = {str(x).strip() for x in pool[id_col].tolist() if str(x).strip()}
    if not cand_ids:
        return pool.reset_index(drop=True)

    use_derived = bool(getattr(cfg, "comp_pool_derived_admission", False))
    from comp_pool_noise import (  # noqa: PLC0415
        CompPoolAdmissionError,
        CompPoolRegime,
        reject_reason_counts,
        write_comp_pool_admission_artifact,
    )
    from invariants_runtime import (  # noqa: PLC0415
        assert_population_nonempty,
    )

    regime = CompPoolRegime.LEGACY
    fail_reason: str | None = None
    admitted_ids: set[str] | None = None
    derived_meta: dict[str, Any] = {}
    decisions_df: pd.DataFrame | None = None
    reason_counts: dict[str, int] = {}
    admission_rules: list[dict[str, Any]] = []

    if use_derived:
        if not per_frame_csv_paths:
            fail_reason = "no_per_frame_csv_paths"
            raise CompPoolAdmissionError(fail_reason)
        try:
            from comp_pool_noise import analyze_draft_comp_pool  # noqa: PLC0415

            proc_dir = Path(per_frame_csv_paths[0]).parent
            draft_id = int(getattr(cfg, "active_draft_id", 0) or 0)
            setup = str(getattr(cfg, "active_setup_name", "") or "")
            gain = float(getattr(cfg, "gain", 1.0) or 1.0)
            rn = float(getattr(cfg, "read_noise", 0.0) or 0.0)
            if not (math.isfinite(gain) and gain > 0):
                gain = 1.0
            if not (math.isfinite(rn) and rn >= 0):
                rn = 0.0
            plate = float(getattr(cfg, "plate_scale_arcsec_per_px", 0.0) or 0.0)
            ap_r = float("nan")
            if "aperture_r_px" in pool.columns:
                ap_r = float(pd.to_numeric(pool["aperture_r_px"], errors="coerce").median())
            ap_arcsec = float(ap_r * plate) if math.isfinite(ap_r) and plate > 0 else None
            gaia_db = str(getattr(cfg, "gaia_db_path", "") or "")
            analysis = analyze_draft_comp_pool(
                proc_dir,
                draft_id=int(draft_id) if draft_id and int(draft_id) > 0 else 0,
                setup=setup or "unknown",
                gain=gain,
                read_noise_e=rn,
                positions=pool if ap_arcsec else None,
                gaia_db_path=gaia_db if ap_arcsec and gaia_db else None,
                aperture_arcsec=ap_arcsec,
            )
            if analysis.get("error"):
                raise CompPoolAdmissionError(str(analysis.get("error")))
            dec = analysis.get("decisions")
            if not (isinstance(dec, pd.DataFrame) and not dec.empty and "admit" in dec.columns):
                raise CompPoolAdmissionError("derived admission returned no decisions table")
            decisions_df = dec
            reason_counts = reject_reason_counts(dec)
            admitted_ids = {
                str(x).strip()
                for x, ok in zip(dec["catalog_id"], dec["admit"])
                if bool(ok) and str(x).strip()
            }
            regime = CompPoolRegime.DERIVED
            derived_meta = {
                "n_admitted": int(analysis.get("n_admitted", 0) or 0),
                "thresholds": analysis.get("thresholds"),
                "scint_vs_sys": analysis.get("scint_vs_sys"),
                "fit_sigma_sys": (analysis.get("fit") or {}).get("sigma_sys_mag"),
            }
            logging.info(
                "[GLOBAL COMP POOL] derived admission: %d admitted of %d summarized",
                int(derived_meta.get("n_admitted", 0)),
                int(analysis.get("n_stars", 0) or 0),
            )
        except CompPoolAdmissionError as _adm_exc:
            if admission_artifact_dir is not None:
                try:
                    write_comp_pool_admission_artifact(
                        Path(admission_artifact_dir) / "comp_pool_admission.json",
                        regime=CompPoolRegime.FAILED,
                        rules=[],
                        reject_reason_counts_map={},
                        fail_reason=str(_adm_exc.reason),
                    )
                except Exception:  # noqa: BLE001
                    pass
            raise
        except Exception as exc:  # noqa: BLE001
            fail_reason = str(exc)
            logging.warning("[GLOBAL COMP POOL] derived admission failed: %s", exc)
            if admission_artifact_dir is not None:
                try:
                    write_comp_pool_admission_artifact(
                        Path(admission_artifact_dir) / "comp_pool_admission.json",
                        regime=CompPoolRegime.FAILED,
                        rules=[],
                        reject_reason_counts_map={},
                        fail_reason=fail_reason,
                    )
                except Exception:  # noqa: BLE001
                    pass
            raise CompPoolAdmissionError(fail_reason) from exc

    # Mutually exclusive by regime, never by bool(admitted_ids).
    apply_rms_prefilter = regime is CompPoolRegime.LEGACY

    relflux_map = compute_global_pool_rms_map(
        cand_ids=cand_ids,
        _masterstars_df=masterstars_df,
        per_frame_csv_paths=per_frame_csv_paths,
        csv_cache=csv_cache,
        flux_col=flux_col,
        min_frames_frac=min_frames_frac,
        edge_bad_frame_frac_max=edge_bad_frame_frac_max,
        max_psf_chi2=max_psf_chi2,
        max_fwhm_factor=max_fwhm_factor,
        fwhm_px=fwhm_px,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        max_comp_rms=max_comp_rms,
        apply_rms_prefilter=apply_rms_prefilter,
    )
    loo_map, _basis = compute_loo_mag_rms_map(
        cand_ids=cand_ids,
        per_frame_csv_paths=per_frame_csv_paths,
        csv_cache=csv_cache,
        flux_col=flux_col,
        min_frames_frac=min_frames_frac,
    )
    pool = attach_comp_rms_to_pool_rows(
        pool,
        loo_map,
        id_col=id_col,
        relflux_map=relflux_map,
        frames_basis=COMP_RMS_FRAMES_BASIS,
    )

    if regime is CompPoolRegime.DERIVED:
        assert admitted_ids is not None
        before = int(len(pool))
        nid = pool[id_col].map(lambda v: str(v).strip() if pd.notna(v) else "")
        if "catalog_id" in pool.columns:
            cid = pool["catalog_id"].map(lambda v: str(v).strip() if pd.notna(v) else "")
            keep = nid.isin(admitted_ids) | cid.isin(admitted_ids)
        else:
            keep = nid.isin(admitted_ids)
        pool = pool.loc[keep].copy()
        thr_payload = derived_meta.get("thresholds")
        admission_rules.append(
            {
                "rule_id": "COMP_POOL_DERIVED_ADMIT",
                "n_in": before,
                "n_out": int(len(pool)),
                "threshold_value": thr_payload,
                "unit": "mixed_derived",
                "regime": regime.value,
            }
        )
        logging.info(
            "[GLOBAL COMP POOL] derived filter: %d -> %d (meta=%s)",
            before,
            int(len(pool)),
            {k: derived_meta.get(k) for k in ("n_admitted", "fit_sigma_sys")},
        )
        # Persist before raise so a failed draft still carries attributable evidence.
        if admission_artifact_dir is not None:
            art = write_comp_pool_admission_artifact(
                Path(admission_artifact_dir) / "comp_pool_admission.json",
                regime=regime,
                rules=admission_rules,
                reject_reason_counts_map=reason_counts,
                fail_reason=fail_reason,
                extra={"derived_meta": {k: derived_meta.get(k) for k in ("n_admitted", "fit_sigma_sys")}},
            )
            if photometry_dir_for_meta is not None:
                try:
                    merge_photometry_pipeline_meta(
                        photometry_dir_for_meta,
                        {
                            "comp_pool_admission": {
                                "artifact": str(art.name),
                                "regime": regime.value,
                                "rules": admission_rules,
                                "reject_reason_counts": reason_counts,
                            }
                        },
                    )
                except Exception as _meta_exc:  # noqa: BLE001
                    logging.debug("[GLOBAL COMP POOL] pipeline_meta stamp failed: %s", _meta_exc)
        assert_population_nonempty(
            n_in=before,
            n_out=int(len(pool)),
            rule_id="COMP_POOL_DERIVED_ADMIT",
            threshold=thr_payload,
            unit="mixed_derived",
            population="stars in global comp pool after static filters",
        )
    elif regime is CompPoolRegime.LEGACY:
        admission_rules.append(
            {
                "rule_id": "COMP_POOL_LEGACY_RMS",
                "n_in": int(len(pool)),
                "n_out": int(len(pool)),
                "threshold_value": float(max_comp_rms),
                "unit": "mag",
                "regime": regime.value,
                "note": "legacy RMS prefilter currently a no-op inside compute_global_pool_rms_map",
            }
        )
        if admission_artifact_dir is not None:
            art = write_comp_pool_admission_artifact(
                Path(admission_artifact_dir) / "comp_pool_admission.json",
                regime=regime,
                rules=admission_rules,
                reject_reason_counts_map={},
                fail_reason=None,
            )
            if photometry_dir_for_meta is not None:
                try:
                    merge_photometry_pipeline_meta(
                        photometry_dir_for_meta,
                        {
                            "comp_pool_admission": {
                                "artifact": str(art.name),
                                "regime": regime.value,
                                "rules": admission_rules,
                            }
                        },
                    )
                except Exception as _meta_exc:  # noqa: BLE001
                    logging.debug("[GLOBAL COMP POOL] pipeline_meta stamp failed: %s", _meta_exc)

    _before_dedupe = int(len(pool))
    pool = _dedupe_comp_pool_by_gaia_key(pool)
    if int(len(pool)) < _before_dedupe:
        logging.info(
            "[GLOBAL COMP POOL] deduped Gaia catalog_id: %d -> %d rows",
            _before_dedupe,
            int(len(pool)),
        )
    logging.info(
        "[GLOBAL COMP POOL] %d kandidatov (z %d masterstars, po filtroch)",
        len(pool),
        len(masterstars_df),
    )
    # P2 determinism: canonical row order
    if "catalog_id" in pool.columns:
        pool = pool.sort_values("catalog_id", kind="mergesort").reset_index(drop=True)
    return pool

def _dedupe_comp_pool_by_gaia_key(pool: pd.DataFrame) -> pd.DataFrame:
    """One row per Gaia ``catalog_id`` (fallback ``name``); keep lowest ``comp_rms`` when duplicated."""
    if pool is None or getattr(pool, "empty", True):
        return pool if pool is not None else pd.DataFrame()
    out = pool.copy()
    id_src = out.get("catalog_id", out.get("name", pd.Series("", index=out.index)))
    out["_gaia_key"] = id_src.map(_normalize_gaia_id)
    out = out[out["_gaia_key"].astype(str).str.strip() != ""]
    if out.empty:
        return out.reset_index(drop=True)
    sort_cols = ["_gaia_key"]
    ascending = [True]
    if "comp_rms" in out.columns:
        sort_cols.append("comp_rms")
        ascending.append(True)
    if "catalog_id" in out.columns:
        sort_cols.append("catalog_id")
        ascending.append(True)
    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    out = out.drop_duplicates(subset=["_gaia_key"], keep="first").drop(columns=["_gaia_key"])
    return out.reset_index(drop=True)

def _warn_zero_compstars_edge(
    *,
    target_cid: str,
    target: pd.Series,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
) -> None:
    """Pri neuspesnom vybere comp (0 riadkov) - ak je ciel blizko vnutorneho okraja cipu, doplni kontext."""
    try:
        tx = float(pd.to_numeric(target.get("x"), errors="coerce"))
        ty = float(pd.to_numeric(target.get("y"), errors="coerce"))
    except Exception:  # noqa: BLE001
        tx = ty = float("nan")
    if not (math.isfinite(tx) and math.isfinite(ty)):
        logging.warning("[COMP] %s: 0 comp stars", target_cid)
        return
    m = int(chip_interior_margin_px)
    if chip_fw is None or chip_fh is None or m <= 0:
        logging.warning("[COMP] %s: 0 comp stars", target_cid)
        return
    wf = float(int(chip_fw))
    hf = float(int(chip_fh))
    if wf <= 2.0 * float(m) or hf <= 2.0 * float(m):
        logging.warning("[COMP] %s: 0 comp stars", target_cid)
        return
    xmin = float(m)
    ymin = float(m)
    xmax = wf - float(m)
    ymax = hf - float(m)
    dist = min(tx - xmin, xmax - tx, ty - ymin, ymax - ty)
    if math.isfinite(dist) and dist < 100.0:
        logging.warning(
            "[COMP] %s: 0 comp stars, target je %.0fpx od okraja bbox "
            "(edge position - geometricky obmedzene pole)",
            target_cid,
            float(dist),
        )
    else:
        logging.warning("[COMP] %s: 0 comp stars", target_cid)

def _count_gate_passing_comps(
    result: pd.DataFrame | None,
    per_target_rms_map: dict[str, float] | None,
    max_comp_rms: float,
    id_col: str,
) -> int:
    """Count comps in ``result`` whose per-target ``comp_rms`` passes the gate.

    N_good = comps passing the colour ladder + per-target ``max_comp_rms`` gate.
    The per-target gate is authoritative: a comp with ``comp_rms > max_comp_rms``
    is never counted as good, so routing never treats an above-gate comp as a
    usable default comp (known-issue (b) fix). When the gate is disabled
    (non-finite / <= 0) fall back to the raw row count.
    """
    if result is None or getattr(result, "empty", True):
        return 0
    if not (math.isfinite(max_comp_rms) and max_comp_rms > 0):
        return int(len(result))
    if id_col not in result.columns:
        return int(len(result))
    _map = per_target_rms_map or {}
    n_good = 0
    for _rid in result[id_col].astype(str).str.strip():
        _v = _map.get(_rid, _map.get(str(_rid), float("nan")))
        try:
            _vf = float(_v)
        except (TypeError, ValueError):
            _vf = float("nan")
        if math.isfinite(_vf) and _vf <= float(max_comp_rms):
            n_good += 1
    return n_good

def select_comparison_stars_per_target(
    target: pd.Series,
    masterstars_df: pd.DataFrame,
    per_frame_csv_paths: list[Path],
    *,
    csv_cache: dict[str, pd.DataFrame] | None = None,
    global_comp_pool_df: pd.DataFrame | None = None,
    fwhm_px: float = 3.7,
    max_dist_deg: float = 1.0,
    max_mag_diff: float = 0.25,  # +-0.25 mag od targetu (zaklad; pri jasnom ciele vid ``mag_tol`` nizsie)
    max_mag_diff_t1: float = 0.50,
    max_mag_diff_t2: float = 1.00,
    max_mag_diff_t3: float = 1.50,
    max_mag_diff_t4: float = 2.00,
    n_comp_min: int = 3,
    n_comp_max: int = 7,
    max_comp_rms: float = 0.1,
    min_dist_arcsec: float = 60.0,
    min_frames_frac: float = 0.3,
    rms_outlier_sigma: float = 3.0,
    exclude_gaia_nss: bool = True,
    exclude_gaia_extobj: bool = True,
    mag_bright_threshold: float = 12.0,
    max_mag_diff_bright_floor: float = 0.0,
    max_psf_chi2: float = 3.0,
    max_fwhm_factor: float = 1.5,
    isolation_radius_px: float = 25.0,
    flux_col: str = "dao_flux",
    chip_fw: int | None = None,
    chip_fh: int | None = None,
    chip_interior_margin_px: int = 0,
    edge_bad_frame_frac_max: float = 0.10,
    max_delta_bprp: float = 0.5,
    vsx_local_db_path: str | None = None,
    gaia_db_path: str | None = None,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
    variable_target_catalog_ids: AbstractSet[str] | None = None,
    cfg: AppConfig | None = None,
    plate_scale_arcsec: float = 1.3,
    use_pixel_dist: bool = False,
    gs11_comp_rejects_acc: list[int] | None = None,
    _selection_mode: str = "auto",
    sat_may_exclude: bool = True,
) -> pd.DataFrame:
    """Faza 1: Pre jeden target vyber najstabilnejsie porovnavacie hviezdy.

    Postup (Moznost D = B + C):
    1. Priestorovy + fotometricky filter kandidatov z masterstars
    2. Nacitaj flux zo vsetkych per-frame CSV (len _PHASE_USECOLS_PERFRAME)
    3. Normalizuj flux voci ensemble medianu per snimka
    4. Vypocitaj RMS scatter pre kazdeho kandidata
    5. Iterativny ensemble filter - vyrad top outlierov kym RMS neklesa
    6. Vrat top n_comp_max najstabilnejsich (min n_comp_min)

    Args:
        exclude_gaia_nss: Vyluc Gaia non-single stars (binarky, vizualne dvojhviezdy).
            Tieto maju variabilny flux nezavisly od pocasia -> scatter comp hviezdy.
        exclude_gaia_extobj: Vyluc Gaia QSO a galaxie (gaia_qso, gaia_gal).
            Nie su bodove zdroje -> systematicke chyby v aperturnej fotometrii.
        max_psf_chi2: Maximalny medianovy PSF chi^2 kandidata cez vsetky snimky.
            Vysoke chi^2 = profil nie je cisty Gaussian = blend alebo rozsireny zdroj.
            Pouzije sa len ak je stlpec psf_chi2 dostupny v per-frame CSV.
            Nastavenie na float("inf") filter vypne.
        max_fwhm_factor: Maximalny pomer fwhm_estimate_px kandidata voci medianu
            vsetkych hviezd na snimke. Hodnota > 1.5 indikuje blend dvoch blizkych
            hviezd. Pouzije sa len ak je stlpec fwhm_estimate_px dostupny.
            Nastavenie na float("inf") filter vypne.
        isolation_radius_px: Polomer v pixeloch pre vypocet contamination indexu.
            Sucet flux susedov / flux kandidata v tomto polomere = contamination.
            Vysledok vstupuje do combined score (soft penalizacia, nie hard exclusion).
            Nastavenie na 0.0 vypne crowding penalizaciu uplne.
        max_comp_rms: Maximalny povoleny p2p RMS scatter comp hviezdy (mag).
            Hviezdy s RMS > max_comp_rms su odmietnute bez ohladu na ranking.
            Default 0.05 mag (50 ppt) - standardna fotometricka stabilita.
        min_dist_arcsec: Minimalna vzdialenost comp hviezdy od targetu v oblukovych
            sekundach. Zabranuje PSF overlap pri velmi blizkych hviezdach.
            Default 60 arcsec (ochrana aj proti lokalnym artefaktom okolo targetu).
        mag_bright_threshold: Hranica ``mag`` ciela (rovnaky system ako ``target["mag"]``),
            pod ktorou sa uplatni ``max_mag_diff_bright_floor`` (typicky jasne hviezdy ~9 mag).
        max_mag_diff_bright_floor: Minimalna sirka |Deltamag| pri jasnych cieloch; ``0`` vypne.
        chip_fw / chip_fh / chip_interior_margin_px: spolu orezu kandidatov na comp hviezdy
            blizko okraja cipu (rovnaka logika ako Faza 0 a suspected). ``chip_interior_margin_px=0`` = vypnute.
        variable_target_catalog_ids: Gaia ``catalog_id`` zo ``variable_targets.csv`` - tieto hviezdy
            sa nikdy neponuknu ako porovnavacky (VSX premenne vratane ``catalog_only``).

    Returns:
        DataFrame s porovnavacimi hviezdami pre tento target, zoradeny podla RMS ASC.
        Prazdny DataFrame ak sa nenajde dostatok stabilnych hviezd.
    """
    from comp_selection_per_target import (  # noqa: PLC0415
        _accumulate_per_frame_comp_metrics,
        _apply_comp_metric_hard_filters,
        _assemble_comp_selection_result_rows,
        _assign_comp_tiers_to_pool,
        _bootstrap_phase1_csv_cache,
        _build_candidates_pre_adaptive_mag,
        _compute_comp_contamination_map,
        _detrend_and_compute_comp_rms_map,
        _ensemble_mad_filter_rms,
        _filter_comp_candidates_spatial_static,
        _iterative_ensemble_clip_cm_residual,
        _resolve_target_color_for_comp_selection,
        _score_comp_candidates_broeg,
    )

    from config import (  # noqa: PLC0415
        resolve_comp_sparse_fallback_enabled,
        resolve_comp_sparse_fallback_min,
    )

    _cfg_p1 = cfg if cfg is not None else AppConfig()
    _mode = str(_selection_mode or "auto").strip().lower()
    if _mode not in ("auto", "default", "sparse_fallback"):
        _mode = "auto"
    sparse_fallback = _mode == "sparse_fallback"

    def _retry_sparse_fallback() -> pd.DataFrame:
        if sparse_fallback or _mode != "auto":
            return pd.DataFrame()
        if not resolve_comp_sparse_fallback_enabled(_cfg_p1):
            return pd.DataFrame()
        return select_comparison_stars_per_target(
            target,
            masterstars_df,
            per_frame_csv_paths,
            csv_cache=csv_cache,
            global_comp_pool_df=global_comp_pool_df,
            fwhm_px=fwhm_px,
            max_dist_deg=max_dist_deg,
            max_mag_diff=max_mag_diff,
            max_mag_diff_t1=max_mag_diff_t1,
            max_mag_diff_t2=max_mag_diff_t2,
            max_mag_diff_t3=max_mag_diff_t3,
            max_mag_diff_t4=max_mag_diff_t4,
            n_comp_min=n_comp_min,
            n_comp_max=n_comp_max,
            max_comp_rms=max_comp_rms,
            min_dist_arcsec=min_dist_arcsec,
            min_frames_frac=min_frames_frac,
            rms_outlier_sigma=rms_outlier_sigma,
            exclude_gaia_nss=exclude_gaia_nss,
            exclude_gaia_extobj=exclude_gaia_extobj,
            mag_bright_threshold=mag_bright_threshold,
            max_mag_diff_bright_floor=max_mag_diff_bright_floor,
            max_psf_chi2=max_psf_chi2,
            max_fwhm_factor=max_fwhm_factor,
            isolation_radius_px=isolation_radius_px,
            flux_col=flux_col,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=chip_interior_margin_px,
            edge_bad_frame_frac_max=edge_bad_frame_frac_max,
            max_delta_bprp=max_delta_bprp,
            vsx_local_db_path=vsx_local_db_path,
            gaia_db_path=gaia_db_path,
            gaia_prefetch=gaia_prefetch,
            variable_target_catalog_ids=variable_target_catalog_ids,
            cfg=cfg,
            plate_scale_arcsec=plate_scale_arcsec,
            use_pixel_dist=use_pixel_dist,
            gs11_comp_rejects_acc=gs11_comp_rejects_acc,
            _selection_mode="sparse_fallback",
        )

    if sparse_fallback:
        ms = masterstars_df.copy()
    elif global_comp_pool_df is not None and not getattr(global_comp_pool_df, "empty", True):
        # COMP-ASSIGN-01 D2: keep step-1 pool ``comp_rms`` (do not drop / re-derive).
        ms = global_comp_pool_df.copy()
    else:
        ms = masterstars_df.copy()
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms.columns:
            ms[_id_col] = _normalize_id_series(ms[_id_col])
    for col in (
        "is_usable",
        "is_saturated",
        "is_noisy",
        "snr50_ok",
        "vsx_known_variable",
        "likely_saturated",
    ):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    ctx = _resolve_target_color_for_comp_selection(
        target,
        vsx_local_db_path=vsx_local_db_path,
        gaia_db_path=gaia_db_path,
        cfg=_cfg_p1,
    )
    target_cid_early = str(ctx["target_cid"])
    try:
        from pinned_ensembles import (  # noqa: PLC0415
            get_pinned_members_for_target,
            select_pinned_comparison_stars_for_target,
        )

        _pin_members = get_pinned_members_for_target(target_cid_early)
    except Exception as _pin_exc:  # noqa: BLE001
        logging.warning("[PIN] pinned ensemble load failed (continuing default): %s", _pin_exc)
        _pin_members = None
    if _pin_members:
        return select_pinned_comparison_stars_for_target(
            target,
            masterstars_df,
            per_frame_csv_paths,
            _pin_members,
            csv_cache=csv_cache,
            fwhm_px=fwhm_px,
            max_dist_deg=max_dist_deg,
            n_comp_min=n_comp_min,
            n_comp_max=n_comp_max,
            max_comp_rms=max_comp_rms,
            min_dist_arcsec=min_dist_arcsec,
            min_frames_frac=min_frames_frac,
            flux_col=flux_col,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
            max_delta_bprp=max_delta_bprp,
            plate_scale_arcsec=float(plate_scale_arcsec),
            use_pixel_dist=bool(use_pixel_dist),
            cfg=_cfg_p1,
            vsx_local_db_path=vsx_local_db_path,
            gaia_db_path=gaia_db_path,
            gaia_prefetch=gaia_prefetch,
        )

    ra_t = float(ctx["ra_t"])
    dec_t = float(ctx["dec_t"])
    mag_t = float(ctx["mag_t"])
    target_cid = str(ctx["target_cid"])
    t_bp_tgt = float(ctx["t_bp_tgt"])
    target_bprp_eff = float(ctx["target_bprp_eff"])
    max_delta_bprp_cfg = float(ctx["max_delta_bprp_cfg"])
    _individual_tier = ctx["_individual_tier"]
    _target_name = str(ctx["_target_name"])

    mag_tol = float(max_mag_diff)
    if (
        math.isfinite(mag_t)
        and float(max_mag_diff_bright_floor) > 0.0
        and mag_t < float(mag_bright_threshold)
    ):
        mag_tol = max(mag_tol, float(max_mag_diff_bright_floor))
        if mag_tol > float(max_mag_diff):
            logging.debug(
                "[FAZA 1] Target %s: jasny ciel (mag=%.2f < %.2f) -> |Deltamag| pas "
                "max(%.3f, floor %.3f) = %.3f",
                target_cid or "?",
                mag_t,
                float(mag_bright_threshold),
                float(max_mag_diff),
                float(max_mag_diff_bright_floor),
                mag_tol,
            )

    _x_t = float(pd.to_numeric(target.get("x"), errors="coerce"))
    _y_t = float(pd.to_numeric(target.get("y"), errors="coerce"))
    ms, _base_mask, det_mask = _filter_comp_candidates_spatial_static(
        ms,
        ra_t=ra_t,
        dec_t=dec_t,
        mag_t=mag_t,
        target_cid=target_cid,
        target_bprp_eff=target_bprp_eff,
        max_delta_bprp_cfg=max_delta_bprp_cfg,
        max_dist_deg=max_dist_deg,
        min_dist_arcsec=min_dist_arcsec,
        exclude_gaia_nss=exclude_gaia_nss,
        exclude_gaia_extobj=exclude_gaia_extobj,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        variable_target_catalog_ids=variable_target_catalog_ids,
        use_pixel_dist=bool(use_pixel_dist),
        x_t=_x_t if math.isfinite(_x_t) else None,
        y_t=_y_t if math.isfinite(_y_t) else None,
        plate_scale_arcsec=float(plate_scale_arcsec),
    )

    built = _build_candidates_pre_adaptive_mag(
        ms,
        _base_mask=_base_mask,
        det_mask=det_mask,
        mag_t=mag_t,
        target_cid=target_cid,
        mag_tol=mag_tol,
        max_mag_diff=max_mag_diff,
        n_comp_min=n_comp_min,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        target=target,
        cfg=_cfg_p1,
        sparse_fallback_mode=sparse_fallback,
        fwhm_dao_px=float(fwhm_px),
        solve_rms_px=None,
    )
    if built is None:
        return _retry_sparse_fallback()
    candidates_pre, used_mag_tol = built

    if str(target_cid).strip() == "1498613634033133184":
        try:
            from comp_selection_per_target import BO_CVN_STEP_COUNTS  # noqa: PLC0415

            BO_CVN_STEP_COUNTS["C_mag_diff"] = int(len(candidates_pre))
        except Exception:  # noqa: BLE001
            # EXC-0203: T3 -- BO CVn BO_CVN_STEP_COUNTS debug counter not updated (EXCEPT-BULK-2 2026-07-08)
            pass

    if str(target_cid).strip() == "1498613634033133184":
        try:
            _dbg = candidates_pre.copy()
            if "_dist_deg" in _dbg.columns and "dist_arcsec" not in _dbg.columns:
                _dbg["dist_arcsec"] = pd.to_numeric(_dbg["_dist_deg"], errors="coerce") * 3600.0
            if "mag" not in _dbg.columns and "_mag" in _dbg.columns:
                _dbg["mag"] = pd.to_numeric(_dbg["_mag"], errors="coerce")
            # Limit columns to the requested view if available
            _cols = [c for c in ["catalog_id", "bp_rp", "mag", "dist_arcsec"] if c in _dbg.columns]
            print(
                f"[DEBUG BO CVn] candidates entering PERF-4B: {int(len(_dbg))} "
                f"(used_mag_tol={float(used_mag_tol):.2f})"
            )
            if _cols:
                print(_dbg[_cols].head(200).to_string(index=False))
        except Exception:  # noqa: BLE001
            # EXC-0204: T3 -- BO CVn candidates debug table print suppressed (EXCEPT-BULK-2 2026-07-08)
            pass

    _r_ap_iso = 7.0
    try:
        _fw = float(fwhm_px)
        if math.isfinite(_fw) and _fw > 0:
            _r_ap_iso = float(2.75 * _fw)
    except (TypeError, ValueError):
        _r_ap_iso = 7.0
    if not (math.isfinite(_r_ap_iso) and _r_ap_iso > 0):
        _r_ap_iso = 7.0
    try:
        # Full catalogue for single-source NN (global pool may omit the neighbour).
        _field_src = masterstars_df if masterstars_df is not None else ms
        ms_arr_x = pd.to_numeric(_field_src.get("x", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        ms_arr_y = pd.to_numeric(_field_src.get("y", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        if "_mag" in _field_src.columns:
            ms_arr_mag = pd.to_numeric(_field_src["_mag"], errors="coerce").to_numpy(dtype=float)
        else:
            ms_arr_mag = pd.to_numeric(_field_src.get("mag", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    except Exception as _iso_exc:  # noqa: BLE001
        logging.warning(f"[FAZA 1] Aperture izolacia preskocena (chyba): {_iso_exc!s}")
        ms_arr_x = ms_arr_y = ms_arr_mag = np.array([], dtype=float)

    id_col = (
        "name"
        if "name" in candidates_pre.columns
        else ("catalog_id" if "catalog_id" in candidates_pre.columns else "name")
    )
    cand_ids = set(candidates_pre[id_col].astype(str).str.strip())

    avail_cols = _PHASE_USECOLS_PERFRAME.copy()
    csv_cache = _bootstrap_phase1_csv_cache(
        per_frame_csv_paths,
        csv_cache,
        flux_col=flux_col,
        avail_cols=avail_cols,
    )
    metrics = _accumulate_per_frame_comp_metrics(
        per_frame_csv_paths,
        csv_cache,
        cand_ids,
        flux_col=flux_col,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
    )
    flux_map = metrics["flux_map"]
    bjd_map = metrics.get("bjd_map") or {}
    n_frames_loaded = int(metrics["n_frames_loaded"])
    psf_chi2_map = metrics["psf_chi2_map"]
    fwhm_map = metrics["fwhm_map"]
    frame_fwhm_medians = metrics["frame_fwhm_medians"]
    peak_over_map = metrics["peak_over_map"]
    peak_total_map = metrics["peak_total_map"]
    snr_map = metrics["snr_map"]
    edge_bad_map = metrics["edge_bad_map"]
    edge_total_map = metrics["edge_total_map"]

    min_frames = max(3, int(n_frames_loaded * min_frames_frac))

    _dilution_map: dict[str, dict[str, Any]] | None = None
    _comp_gs11_notes: dict[str, str] = {}
    if bool(_cfg_p1.gs11_dilution_enabled) and gaia_db_path:
        from dilution import compute_dilution_factor  # noqa: PLC0415

        _ap_cfg = float(_cfg_p1.gs11_dilution_aperture_arcsec)
        if math.isfinite(_ap_cfg) and _ap_cfg > 0:
            _ap_arcsec = _ap_cfg
        else:
            _ap_r_px = 2.75 * float(fwhm_px) if math.isfinite(float(fwhm_px)) and float(fwhm_px) > 0 else 7.0
            _ap_arcsec = float(_ap_r_px) * float(plate_scale_arcsec)
        _dilution_map = {}
        for _, crow in candidates_pre.iterrows():
            _cid_d = str(crow.get(id_col, crow.get("catalog_id", "")) or "").strip()
            if not _cid_d:
                continue
            try:
                _ra_d = float(pd.to_numeric(crow.get("ra_deg", crow.get("ra")), errors="coerce"))
                _dec_d = float(pd.to_numeric(crow.get("dec_deg", crow.get("dec")), errors="coerce"))
            except (TypeError, ValueError):
                continue
            _gm_d = float("nan")
            for _gcol in ("phot_g_mean_mag", "mag", "_mag"):
                if _gcol in crow.index:
                    try:
                        _gv = float(pd.to_numeric(crow[_gcol], errors="coerce"))
                    except (TypeError, ValueError):
                        _gv = float("nan")
                    if math.isfinite(_gv):
                        _gm_d = _gv
                        break
            from dilution import _normalize_exclude_source_id  # noqa: PLC0415

            _dilution_map[_cid_d] = compute_dilution_factor(
                _ra_d,
                _dec_d,
                _gm_d,
                _ap_arcsec,
                str(gaia_db_path),
                catalog_id=_normalize_exclude_source_id(_cid_d),
                mag_limit_delta=float(_cfg_p1.gs11_dilution_mag_limit_delta),
            )

    flux_map, _b_rejected = _apply_comp_metric_hard_filters(
        flux_map,
        peak_over_map,
        peak_total_map,
        snr_map,
        psf_chi2_map,
        fwhm_map,
        frame_fwhm_medians,
        edge_bad_map,
        edge_total_map,
        target_cid=target_cid,
        edge_bad_frame_frac_max=edge_bad_frame_frac_max,
        max_psf_chi2=max_psf_chi2,
        max_fwhm_factor=max_fwhm_factor,
        dilution_map=_dilution_map,
        cfg=_cfg_p1,
        comp_quality_notes=_comp_gs11_notes,
        sat_may_exclude=bool(sat_may_exclude),
    )
    if gs11_comp_rejects_acc is not None and _dilution_map:
        _max_d_gs11 = float(_cfg_p1.gs11_comp_max_dilution)
        for _cid_r in _b_rejected:
            _ent = _dilution_map.get(str(_cid_r), {})
            try:
                _d_r = float(_ent.get("dilution_factor", 1.0))
            except (TypeError, ValueError):
                _d_r = 1.0
            if math.isfinite(_d_r) and _d_r < _max_d_gs11:
                gs11_comp_rejects_acc[0] += 1

    contamination_map = _compute_comp_contamination_map(
        flux_map,
        ms,
        target_cid=target_cid,
        isolation_radius_px=isolation_radius_px,
    )

    _use_iter_clip = bool(sparse_fallback)

    # COMP-ASSIGN-01 D2: one RMS measurement - prefer pool step-1 ``comp_rms``.
    _id_rms_col = (
        "catalog_id"
        if "catalog_id" in ms.columns
        else ("name" if "name" in ms.columns else None)
    )
    _pool_rms_map: dict[str, float] = {}
    if (not sparse_fallback) and _id_rms_col and "comp_rms" in ms.columns:
        _ids = ms[_id_rms_col].map(lambda v: str(v).strip() if pd.notna(v) else "")
        _vals = pd.to_numeric(ms["comp_rms"], errors="coerce")
        for _cid_p, _rv_p in zip(_ids.tolist(), _vals.tolist(), strict=False):
            if _cid_p and math.isfinite(float(_rv_p)):
                _pool_rms_map[str(_cid_p)] = float(_rv_p)
    if _pool_rms_map and not _use_iter_clip:
        rms_map = dict(_pool_rms_map)
        sorted_rms_map = dict(
            sorted(rms_map.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
        )
        logging.info(
            "[COMP-ASSIGN] Target %s: using pool step-1 comp_rms (n=%d); "
            "skip per-target RMS re-derivation",
            target_cid,
            len(rms_map),
        )
    else:
        rms_result = _detrend_and_compute_comp_rms_map(
            flux_map,
            min_frames=min_frames,
            max_comp_rms=max_comp_rms,
            n_comp_min=n_comp_min,
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
            skip_apriori_rms=_use_iter_clip,
        )
        if rms_result[0] is None:
            return _retry_sparse_fallback()
        rms_map, sorted_rms_map = rms_result

    def _apply_aperture_isolation_safe(cands: pd.DataFrame) -> pd.DataFrame:
        if cands.empty:
            return cands
        try:
            ms_arr_x2 = ms_arr_x
            ms_arr_y2 = ms_arr_y
            ms_arr_mag2 = ms_arr_mag
        except Exception as exc:  # noqa: BLE001
            # EXC-0206: T3 -- BO CVn RMS-rejection funnel debug list not built (EXCEPT-BULK-2 2026-07-08)
            logging.error('[EXC-0205] Aperture isolation filter skipped when ms_arr arrays unavailable - crowded comps not re...: %s', exc)
            return cands
        rej: set[Any] = set()
        for idx2, crow2 in cands.iterrows():
            cx2 = float(crow2.get("x", float("nan")))
            cy2 = float(crow2.get("y", float("nan")))
            cm2 = float(crow2.get("_mag", float("nan"))) if "_mag" in cands.columns else float("nan")
            if not (math.isfinite(cx2) and math.isfinite(cy2) and math.isfinite(cm2)):
                continue
            d2 = np.sqrt((ms_arr_x2 - cx2) ** 2 + (ms_arr_y2 - cy2) ** 2)
            in_ap2 = (d2 < float(_r_ap_iso)) & (d2 > 1e-6)
            if not bool(np.any(in_ap2)):
                continue
            nm2 = ms_arr_mag2[in_ap2]
            sig2 = nm2[np.isfinite(nm2) & (np.abs(nm2 - cm2) < 3.0)]
            if int(sig2.size) > 0:
                rej.add(idx2)
        if not rej:
            return cands
        after = int(len(cands) - len(rej))
        if after >= int(n_comp_min):
            return cands[~cands.index.isin(rej)]
        return cands

    candidates = ms[_base_mask | det_mask].copy()
    if candidates.empty:
        logging.warning(f"[FAZA 1] {target_cid}: ziadni kandidati po hard filtroch")
        _warn_zero_compstars_edge(
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        return _retry_sparse_fallback()
    candidates = _apply_aperture_isolation_safe(candidates)

    clip_meta: dict[str, int] | None = None
    if _use_iter_clip:
        _clip_out = _iterative_ensemble_clip_cm_residual(
            flux_map,
            bjd_map,
            sorted_rms_map,
            clip_sigma=5.0,
            n_comp_min=n_comp_min,
            min_final=1 if sparse_fallback else None,
        )
        if _clip_out is None:
            return pd.DataFrame()
        active, clip_meta = _clip_out
    else:
        active = _ensemble_mad_filter_rms(
            rms_map,
            candidates,
            target_cid=target_cid,
            target=target,
            n_comp_min=n_comp_min,
            rms_outlier_sigma=rms_outlier_sigma,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
    if active is None:
        return _retry_sparse_fallback()

    _bo_funnel: dict[str, int] = {}
    _bo_rms_rejected: list[tuple[str, float]] = []
    if str(target_cid).strip() == "1498613634033133184":
        try:
            _bo_funnel["F_perf4b"] = int(len(candidates_pre))
            _bo_funnel["G_after_rms"] = int(len(active))
            for _cid_r, _rv in sorted(
                (sorted_rms_map or {}).items(), key=lambda kv: (float(kv[1]), str(kv[0]))
            ):
                if _cid_r not in active:
                    if math.isfinite(float(_rv)) and float(_rv) > float(max_comp_rms):
                        _bo_rms_rejected.append((str(_cid_r), float(_rv)))
        except Exception:  # noqa: BLE001
            # EXC-0207: T3 -- BO CVn comp funnel summary log not emitted (EXCEPT-BULK-2 2026-07-08)
            pass

    id_col_cand = (
        "name"
        if "name" in candidates.columns
        else ("catalog_id" if "catalog_id" in candidates.columns else "name")
    )
    score_map, tier_map = _score_comp_candidates_broeg(
        active,
        candidates,
        contamination_map,
        id_col_cand=id_col_cand,
        mag_t=mag_t,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=t_bp_tgt,
        _individual_tier=_individual_tier,
        cfg=_cfg_p1,
    )

    tier_out = _assign_comp_tiers_to_pool(
        candidates,
        active,
        id_col_cand=id_col_cand,
        target=target,
        target_cid=target_cid,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=t_bp_tgt,
        mag_t=mag_t,
        _individual_tier=_individual_tier,
        _target_name=_target_name,
        max_mag_diff_t1=max_mag_diff_t1,
        max_mag_diff=max_mag_diff,
        gaia_db_path=gaia_db_path,
        vsx_local_db_path=vsx_local_db_path,
        gaia_prefetch=gaia_prefetch,
        n_comp_min=n_comp_min,
        n_comp_max=n_comp_max,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        cfg=_cfg_p1,
        max_comp_rms=float(max_comp_rms),
        fwhm_px=float(fwhm_px) if fwhm_px is not None else None,
        field_x=ms_arr_x,
        field_y=ms_arr_y,
    )
    final_comps = tier_out["final_comps"]
    if final_comps is None or getattr(final_comps, "empty", True):
        return _retry_sparse_fallback()

    if str(target_cid).strip() == "1498613634033133184":
        try:
            from comp_selection_per_target import (  # noqa: PLC0415
                _log_bo_cvn_comp_funnel,
                bo_cvn_funnel_snapshot,
            )

            _bo_funnel.update(bo_cvn_funnel_snapshot())
            _bo_funnel["H_after_n_comp_max"] = int(len(final_comps))
            _bo_funnel["final_selected"] = int(len(final_comps))
            _log_bo_cvn_comp_funnel(
                step_counts=_bo_funnel,
                max_comp_rms=float(max_comp_rms),
                n_comp_max=int(n_comp_max),
                rms_rejected=_bo_rms_rejected,
            )
        except Exception:  # noqa: BLE001
            pass

    try:
        final_lookup = final_comps.copy()
        final_lookup[id_col_cand] = final_lookup[id_col_cand].astype(str).str.strip()
        final_lookup = final_lookup.set_index(id_col_cand, drop=False)
    except Exception:  # noqa: BLE001
        final_lookup = None

    result = _assemble_comp_selection_result_rows(
        tier_out["selected_ids"],
        final_comps,
        id_col_cand=id_col_cand,
        active=active,
        score_map=score_map,
        contamination_map=contamination_map,
        flux_map=flux_map,
        target_cid=target_cid,
        target=target,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=t_bp_tgt,
        sel_note=str(tier_out["sel_note"]),
        used_mag_tol=float(used_mag_tol),
        best_tier=str(tier_out["best_tier"]),
        tier4_warning=bool(tier_out["tier4_warning"]),
        n_t1=int(tier_out["n_t1"]),
        n_t2=int(tier_out["n_t2"]),
        n_t3=int(tier_out["n_t3"]),
        n_t4=int(tier_out["n_t4"]),
        comp_bprp_map=tier_out["comp_bprp_map"],
        comp_tier_final_map=tier_out["comp_tier_final_map"],
        comp_delta_bprp_map=tier_out["comp_delta_bprp_map"],
        comp_color_tier_src_map=tier_out["comp_color_tier_src_map"],
        _b_rejected=_b_rejected,
        final_lookup=final_lookup,
        dilution_map=_dilution_map,
        comp_gs11_notes=_comp_gs11_notes,
        cfg=_cfg_p1,
        clip_meta=clip_meta,
        comp_path="sparse_fallback" if sparse_fallback else "default",
        per_target_rms_map=rms_map,
    )

    if _mode == "auto":
        # Route on the count of comps passing the per-target comp_rms gate, not raw
        # len(result): zero gate-passers -> sparse_fallback (known-issue (b) fix).
        _n_good = _count_gate_passing_comps(result, rms_map, max_comp_rms, id_col_cand)
        if _n_good >= 1:
            return result
        if resolve_comp_sparse_fallback_enabled(_cfg_p1):
            fb = select_comparison_stars_per_target(
                target,
                masterstars_df,
                per_frame_csv_paths,
                csv_cache=csv_cache,
                global_comp_pool_df=global_comp_pool_df,
                fwhm_px=fwhm_px,
                max_dist_deg=max_dist_deg,
                max_mag_diff=max_mag_diff,
                max_mag_diff_t1=max_mag_diff_t1,
                max_mag_diff_t2=max_mag_diff_t2,
                max_mag_diff_t3=max_mag_diff_t3,
                max_mag_diff_t4=max_mag_diff_t4,
                n_comp_min=n_comp_min,
                n_comp_max=n_comp_max,
                max_comp_rms=max_comp_rms,
                min_dist_arcsec=min_dist_arcsec,
                min_frames_frac=min_frames_frac,
                rms_outlier_sigma=rms_outlier_sigma,
                exclude_gaia_nss=exclude_gaia_nss,
                exclude_gaia_extobj=exclude_gaia_extobj,
                mag_bright_threshold=mag_bright_threshold,
                max_mag_diff_bright_floor=max_mag_diff_bright_floor,
                max_psf_chi2=max_psf_chi2,
                max_fwhm_factor=max_fwhm_factor,
                isolation_radius_px=isolation_radius_px,
                flux_col=flux_col,
                chip_fw=chip_fw,
                chip_fh=chip_fh,
                chip_interior_margin_px=chip_interior_margin_px,
                edge_bad_frame_frac_max=edge_bad_frame_frac_max,
                max_delta_bprp=max_delta_bprp,
                vsx_local_db_path=vsx_local_db_path,
                gaia_db_path=gaia_db_path,
                gaia_prefetch=gaia_prefetch,
                variable_target_catalog_ids=variable_target_catalog_ids,
                cfg=cfg,
                plate_scale_arcsec=plate_scale_arcsec,
                use_pixel_dist=use_pixel_dist,
                gs11_comp_rejects_acc=gs11_comp_rejects_acc,
                _selection_mode="sparse_fallback",
            )
            _n_fb = int(len(fb)) if fb is not None and not getattr(fb, "empty", True) else 0
            if _n_fb >= 1:
                return fb
        return pd.DataFrame()

    return result

def _write_suspected_variables(
    ms_df: pd.DataFrame,
    csv_paths: list[Path],
    active_target_ids: set[str],
    output_path: Path,
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float = 0.5,
    outlier_sigma: float = 3.0,
    interior_fw: int | None = None,
    interior_fh: int | None = None,
    interior_margin_px: int | None = None,
    csv_cache: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Detekuj hviezdy s vysokym RMS scatter ktore nie su v VSX - suspected new variables.

    Zapise suspected_variables.csv s kolumnami:
    catalog_id, ra_deg, dec_deg, mag, comp_rms, n_frames, zone

    Ak su zadane ``interior_*``, vyhodi sa pool aj per-frame body pri okrajoch cipu
    (rovnaky okraj ako pri aktivnych cieloch a porovnavackach vo ``run_phase0_and_phase1``).
    """
    # Usable hviezdy ktore nie su VSX ani active targets
    ms = ms_df.copy()
    for col in ("is_usable", "is_saturated", "is_noisy", "vsx_known_variable"):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    id_col = "catalog_id" if "catalog_id" in ms.columns else "name"
    base_mask = (
        _bool_col(ms.get("is_usable", pd.Series(True, index=ms.index)))
        & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("is_noisy", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
    )
    pool = ms[base_mask].copy()
    pool["_nid"] = pool[id_col].map(_normalize_id_value)
    pool = pool[pool["_nid"] != ""].drop_duplicates(subset=["_nid"], keep="first")

    _m = int(interior_margin_px) if interior_margin_px is not None else 0
    _fw = int(interior_fw) if interior_fw is not None else 0
    _fh = int(interior_fh) if interior_fh is not None else 0
    if (
        _m > 0
        and _fw > 2 * _m
        and _fh > 2 * _m
        and "x" in pool.columns
        and "y" in pool.columns
    ):
        _xn = pd.to_numeric(pool["x"], errors="coerce")
        _yn = pd.to_numeric(pool["y"], errors="coerce")
        _ok = _xn.between(_m, _fw - _m) & _yn.between(_m, _fh - _m)
        _n_pool0 = int(len(pool))
        pool = pool[_ok].copy()
        logging.info(
            "[SUSPECTED] Orezanie okrajov (rovnake ako Faza 0/1, MASTERSTAR x,y): %s -> %s hviezd (margin %s px, pole %sx%s)",
            _n_pool0,
            len(pool),
            _m,
            _fw,
            _fh,
        )

    pool_ids = set(pool["_nid"]) - active_target_ids

    if not pool_ids:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Nacitaj flux pre vsetky hviezdy z poolu
    flux_map: dict[str, list[float]] = {cid: [] for cid in pool_ids}
    n_frames = 0
    _cache_hits = 0
    _cache_misses = 0

    for csv_path in csv_paths:
        try:
            _cache_key = str(csv_path)
            _cached = csv_cache.get(_cache_key) if csv_cache else None
            if _cached is not None and not _cached.empty:
                header_cols = _cached.columns
            else:
                header_cols = pd.read_csv(csv_path, nrows=0).columns
            actual_flux = flux_col if flux_col in header_cols else "flux"
            name_c = "catalog_id" if "catalog_id" in header_cols else "name"
            use = [name_c, actual_flux]
            if "mag" in header_cols and "mag" not in use:
                use.append("mag")
            _use_xy = _m > 0 and _fw > 2 * _m and _fh > 2 * _m
            if _use_xy and "x" in header_cols and "y" in header_cols:
                use.extend([c for c in ("x", "y") if c not in use])
            if _cached is not None and not _cached.empty:
                df = _cached[[c for c in use if c in _cached.columns]].copy()
                _cache_hits += 1
            else:
                df = read_vyvar_csv(csv_path, usecols=use, low_memory=False)
                _cache_misses += 1
            if name_c not in df.columns:
                continue
            df[name_c] = _normalize_id_series(df[name_c])
            df[actual_flux] = pd.to_numeric(df[actual_flux], errors="coerce")
            sub = df[df[name_c].isin(pool_ids) & df[actual_flux].gt(0)]
            if _use_xy and "x" in sub.columns and "y" in sub.columns:
                _xs = pd.to_numeric(sub["x"], errors="coerce")
                _ys = pd.to_numeric(sub["y"], errors="coerce")
                sub = sub[_xs.between(_m, _fw - _m) & _ys.between(_m, _fh - _m)]
            if sub.empty:
                continue

            # Mag-bin normalizacia: median zvlast pre kazdy mag bin (0.5 mag sirka)
            mag_col_frame = "mag" if "mag" in df.columns else None
            if mag_col_frame and mag_col_frame in sub.columns:
                sub = sub.copy()
                sub["_mag_num"] = pd.to_numeric(sub[mag_col_frame], errors="coerce")
                sub["_mag_bin"] = (sub["_mag_num"] / 0.5).apply(
                    lambda x: int(x) if math.isfinite(x) else -1
                )
                bin_meds: dict[int, float] = {}
                for b, grp in sub.groupby("_mag_bin"):
                    bmed = float(grp[actual_flux].median())
                    if math.isfinite(bmed) and bmed > 0:
                        bin_meds[int(b)] = bmed
                if not bin_meds:
                    continue
            else:
                # Fallback: globalny median
                frame_med = float(sub[actual_flux].median())
                if not math.isfinite(frame_med) or frame_med <= 0:
                    continue
                bin_meds = {}

            n_frames += 1
            # Jedna vzorka na hviezdu na snimok (CSV moze mat duplicitne riadky).
            _agg: dict[str, dict[str, float]] = {}
            for _, row in sub.iterrows():
                cid = str(row[name_c])
                if cid not in pool_ids:
                    continue
                raw_flux = float(row[actual_flux])
                if not math.isfinite(raw_flux) or raw_flux <= 0:
                    continue
                mag_num = (
                    float(row.get("_mag_num", float("nan")))
                    if "_mag_num" in row.index
                    else float("nan")
                )
                ent = _agg.setdefault(cid, {"fluxes": [], "mags": []})
                ent["fluxes"].append(raw_flux)
                if math.isfinite(mag_num):
                    ent["mags"].append(mag_num)
            for cid, ent in _agg.items():
                fluxes = ent["fluxes"]
                if not fluxes:
                    continue
                raw_flux = float(np.median(np.asarray(fluxes, dtype=np.float64)))
                if not math.isfinite(raw_flux) or raw_flux <= 0:
                    continue
                mags = ent["mags"]
                mag_num = float(np.median(np.asarray(mags, dtype=np.float64))) if mags else float("nan")
                if bin_meds:
                    b = int(mag_num / 0.5) if math.isfinite(mag_num) else -1
                    norm_med = bin_meds.get(b)
                    if norm_med is None:
                        closest = min(bin_meds.keys(), key=lambda k: abs(k - b))
                        norm_med = bin_meds[closest]
                else:
                    norm_med = frame_med  # type: ignore[assignment]
                rel = raw_flux / norm_med
                if math.isfinite(rel) and rel > 0:
                    flux_map[cid].append(rel)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0222] One frame skipped in suspected-variables flux accumulation - star RMS computed without ...: %s', exc)
            continue

    logging.info(
        "[PERF-1] _write_suspected_variables: %d cache hits, %d disk reads (of %d frames)",
        _cache_hits,
        _cache_misses,
        len(csv_paths),
    )
    if _cache_misses > 0:
        logging.warning(
            "[PERF-1] %d frames read from disk (not in shared_csv_cache) - "
            "check if csv_cache is populated before calling _write_suspected_variables",
            _cache_misses,
        )

    # Airmass detrending pre suspected variables
    for cid in list(flux_map.keys()):
        vals = flux_map[cid]
        if len(vals) < 6:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        t = np.linspace(0.0, 1.0, len(arr))
        try:
            coeffs = _safe_polyfit(t, arr, 2)
            if coeffs is None:
                continue
            trend_fit = np.polyval(coeffs, t)
            safe_trend = np.where(np.abs(trend_fit) > 1e-9, trend_fit, 1.0)
            detrended = arr / safe_trend
            med_dt = float(np.median(detrended))
            if math.isfinite(med_dt) and med_dt > 0:
                flux_map[cid] = (detrended / med_dt).tolist()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0223] Detrend fit failure in suspected-variables leaves raw flux - false variable candidates ...: %s', exc)
            pass

    min_f = max(3, int(n_frames * min_frames_frac))
    rms_map: dict[str, float] = {}
    nframes_map: dict[str, int] = {}
    for cid, vals in flux_map.items():
        if len(vals) < min_f:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        if math.isfinite(rms):
            rms_map[cid] = rms
            nframes_map[cid] = len(vals)

    # COMP-RMS-DEF-01-B: flag on LOO mag MAD; keep mag-bin relflux RMS as diagnostic.
    _csv_cache = csv_cache or {}
    loo_map, _basis = compute_loo_mag_rms_map(
        set(pool_ids),
        csv_paths,
        _csv_cache,
        flux_col=flux_col,
        min_frames_frac=min_frames_frac,
    )
    if not loo_map:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    _MAD_CONSISTENCY = 0.6745
    loo_arr = np.asarray(list(loo_map.values()), dtype=np.float64)
    med = float(np.median(loo_arr))
    mad_raw = float(np.median(np.abs(loo_arr - med)))
    if not math.isfinite(mad_raw) or mad_raw <= 0:
        mad_sigma = float(np.std(loo_arr)) / _MAD_CONSISTENCY or 1e-9
    else:
        mad_sigma = mad_raw / _MAD_CONSISTENCY
    threshold = med + outlier_sigma * mad_sigma

    suspected = {cid: rms for cid, rms in loo_map.items() if rms > threshold}

    if not suspected:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    rows = []
    pool_idx = pool.set_index("_nid", drop=False)
    for cid, rms in sorted(suspected.items(), key=lambda x: -x[1]):
        if cid not in pool_idx.index:
            continue
        r = pool_idx.loc[cid]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append(
            {
                "catalog_id": cid,
                "ra_deg": r.get("ra_deg", float("nan")),
                "dec_deg": r.get("dec_deg", float("nan")),
                "mag": r.get("mag", float("nan")),
                "comp_rms_loo_mag": rms,
                "comp_relflux_mad": float(rms_map.get(cid, float("nan"))),
                "comp_rms": rms,
                "n_frames": nframes_map.get(cid, 0),
                "zone": r.get("zone", ""),
            }
        )

    out_df = pd.DataFrame(rows)
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in out_df.columns:
            out_df["catalog_id"] = normalize_gaia_source_id_series(out_df["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0224] suspected_variables.csv catalog_id normalization fails - exported IDs may be float-corr...: %s', exc)
        pass
    out_df.to_csv(output_path, index=False)
    logging.info(
        f"[SUSPECTED] {len(out_df)} kandidatov na nove premenne -> {output_path.name} "
        f"(threshold RMS > {threshold:.4f})"
    )
