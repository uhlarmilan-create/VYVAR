"""Streamlit UI helpers for calibration / smart-import (equipment header, master binning, multi-obs status)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from calibration import read_master_binning_from_fits
from database import VyvarDatabase
from importer import observation_group_folder_name


def render_calibration_equipment_header(
    db: VyvarDatabase,
    *,
    draft_id: int | None,
    equipment_id: int | None,
    telescope_id: int | None,
) -> None:
    """Top-of-section info: telescope + sensor from OBS_DRAFT JOIN, else from selected IDs."""
    telescope_name: str | None = None
    telescope_focal_mm: float | None = None
    equipment_name: str | None = None
    pixel_um: float | None = None

    if draft_id is not None and int(draft_id) > 0:
        row = db.fetch_obs_draft_telescope_equipment(int(draft_id))
        if row:
            telescope_name = row.get("telescope_name")
            telescope_focal_mm = row.get("telescope_focal_mm")
            equipment_name = row.get("equipment_name")
            pixel_um = row.get("pixel_um")

    if equipment_name is None and equipment_id is not None:
        r = db.conn.execute(
            "SELECT CAMERANAME, PIXELSIZE FROM EQUIPMENTS WHERE ID = ?;",
            (int(equipment_id),),
        ).fetchone()
        if r is not None:
            equipment_name = r["CAMERANAME"]
            pixel_um = r["PIXELSIZE"]

    if telescope_name is None and telescope_id is not None:
        r = db.conn.execute(
            "SELECT TELESCOPENAME, FOCAL FROM TELESCOPE WHERE ID = ?;",
            (int(telescope_id),),
        ).fetchone()
        if r is not None:
            telescope_name = r["TELESCOPENAME"]
            telescope_focal_mm = r["FOCAL"]

    tn = str(telescope_name or "—").strip() or "—"
    en = str(equipment_name or "—").strip() or "—"
    try:
        tf = float(telescope_focal_mm) if telescope_focal_mm is not None else None
    except (TypeError, ValueError):
        tf = None
    try:
        pu = float(pixel_um) if pixel_um is not None else None
    except (TypeError, ValueError):
        pu = None

    focal_s = f"{tf:g} mm" if tf is not None and math.isfinite(tf) else "—"
    pix_s = f"{pu:g} µm" if pu is not None and math.isfinite(pu) else "—"

    st.info(
        f"**Teleskop:** {tn} ({focal_s})\n\n**Senzor:** {en} ({pix_s})",
        icon="🔭",
    )


def render_calibration_library_flat_warnings(db: VyvarDatabase, plan: Any) -> None:
    """Warn when a detected filter has no Master Flat row in CALIBRATION_LIBRARY (SQL)."""
    filters = list(getattr(plan, "detected_filters", None) or [])
    if not filters:
        og = getattr(plan, "observation_groups", None) or {}
        filters = sorted({str(g.get("filter") or "NoFilter") for g in og.values()})
    seen: set[str] = set()
    for raw in filters:
        fl = str(raw or "").strip() or "NoFilter"
        if fl in seen:
            continue
        seen.add(fl)
        if db.calibration_library_has_flat_for_filter(fl):
            continue
        st.warning(
            f"Filter **{fl}** nemá Master v knižnici. Bude použitý fallback alebo preskočené.",
            icon="⚠️",
        )


def _read_bin_safe(p: str | Path | None) -> int | None:
    if not p:
        return None
    pp = Path(str(p))
    if not pp.is_file():
        return None
    try:
        return int(read_master_binning_from_fits(pp))
    except Exception:  # noqa: BLE001
        return None


def build_master_calibration_files_dataframe(plan: Any) -> pd.DataFrame:
    """Per observation group: paths + **Binning Mode** (e.g. Resampling 1x1 -> 2x2)."""
    og = getattr(plan, "observation_groups", None) or {}
    if not og:
        return pd.DataFrame(
            columns=[
                "Skupina",
                "Filter",
                "Exp (s)",
                "Light bin",
                "Master dark",
                "Master flat",
                "Binning Mode",
            ]
        )

    mf_by_obs = getattr(plan, "masterflat_by_obs_key", None) or {}
    md_by_obs = getattr(plan, "dark_master_by_obs_key", None) or {}
    rows: list[dict[str, Any]] = []

    for gk, g in sorted(og.items(), key=lambda x: x[0]):
        lb = max(1, int(g.get("binning") or 1))
        mdp = md_by_obs.get(gk)
        mfp = mf_by_obs.get(gk)
        d_bin = _read_bin_safe(mdp)
        f_bin = _read_bin_safe(mfp)

        parts: list[str] = []
        if d_bin is not None and lb > d_bin and lb % d_bin == 0:
            parts.append(f"Resampling {d_bin}x{d_bin} -> {lb}x{lb} (dark)")
        elif d_bin is not None and d_bin != lb:
            parts.append(f"Dark: master {d_bin}x{d_bin} vs light {lb}x{lb}")

        if f_bin is not None and lb > f_bin and lb % f_bin == 0:
            parts.append(f"Resampling {f_bin}x{f_bin} -> {lb}x{lb} (flat)")
        elif f_bin is not None and f_bin != lb:
            parts.append(f"Flat: master {f_bin}x{f_bin} vs light {lb}x{lb}")

        if not parts:
            if not mfp and not mdp:
                mode = "Bez priradeného dark/flat"
            elif (d_bin == lb or mdp is None) and (f_bin == lb or mfp is None):
                mode = "Priame párovanie"
            else:
                mode = "—"
        else:
            mode = "; ".join(parts)

        rows.append(
            {
                "Skupina": gk,
                "Filter": g.get("filter"),
                "Exp (s)": g.get("exposure_s"),
                "Light bin": lb,
                "Master dark": str(mdp) if mdp else "",
                "Master flat": str(mfp) if mfp else "",
                "Binning Mode": mode,
            }
        )

    return pd.DataFrame(rows)


def _group_keys_by_filter_exp(plan: Any) -> dict[tuple[str, float], list[str]]:
    og = getattr(plan, "observation_groups", None) or {}
    out: dict[tuple[str, float], list[str]] = {}
    for gk, g in og.items():
        f = str(g.get("filter") or "NoFilter")
        try:
            e = float(g.get("exposure_s", 0.0))
        except (TypeError, ValueError):
            e = 0.0
        out.setdefault((f, e), []).append(gk)
    return out


def _count_calibrated_fits(archive_path: Path, group_keys: list[str]) -> int:
    ap = Path(archive_path)
    n = 0
    for gk in group_keys:
        rel = observation_group_folder_name(gk)
        cal_dir = ap / "calibrated" / "lights" / rel
        if not cal_dir.is_dir():
            continue
        for pat in ("*.fits", "*.fit", "*.fts"):
            n += len(list(cal_dir.glob(pat)))
    return n


def _plate_solve_ok_for_groups(archive_path: Path, group_keys: list[str]) -> bool:
    ap = Path(archive_path)
    if not group_keys:
        return False
    root_ms = ap / "platesolve" / "MASTERSTAR.fits"
    for gk in group_keys:
        rel = observation_group_folder_name(gk)
        sub_ms = ap / "platesolve" / rel / "MASTERSTAR.fits"
        if sub_ms.is_file():
            continue
        if len(group_keys) == 1 and root_ms.is_file():
            continue
        return False
    return True


def _preview_calibration_status(plan: Any, gks: list[str]) -> str:
    mf_by_obs = getattr(plan, "masterflat_by_obs_key", None) or {}
    md_by_obs = getattr(plan, "dark_master_by_obs_key", None) or {}
    flat_ok = all(bool(mf_by_obs.get(gk)) for gk in gks)
    dark_ok = all(bool(md_by_obs.get(gk)) for gk in gks)
    if getattr(plan, "quick_look", False):
        return "Quick look (bez vhodného master dark)"
    if flat_ok and dark_ok:
        return "Pripravené na kalibráciu"
    if not flat_ok:
        return "Chýba Master Flat (skupina)"
    return "Chýba Master Dark (skupina)"


def build_multi_observation_status_dataframe(
    plan: Any,
    *,
    archive_path: Path | str | None = None,
    cal_phase: str = "preview",
) -> pd.DataFrame:
    """One row per (Filter, Exp): calibration + plate-solve status columns."""
    combos = _group_keys_by_filter_exp(plan)
    if not combos:
        return pd.DataFrame(
            columns=["Filter", "Exp (s)", "Status Calibration", "Status Plate Solve"],
        )

    ap = Path(archive_path) if archive_path else None
    rows_out: list[dict[str, Any]] = []

    for (f, e) in sorted(combos.keys(), key=lambda x: (x[0], x[1])):
        gks = combos[(f, e)]
        if cal_phase == "running":
            cal_st = "Prebieha…"
            ps_st = "—"
        elif ap is not None and ap.is_dir() and cal_phase == "done":
            n_cal = _count_calibrated_fits(ap, gks)
            cal_st = f"Hotovo ({n_cal} FITS)" if n_cal > 0 else "Bez calibrated / draft"
            ps_st = "Hotovo (MASTERSTAR)" if _plate_solve_ok_for_groups(ap, gks) else "Čaká / nie je"
        else:
            cal_st = _preview_calibration_status(plan, gks)
            ps_st = "Po kalibrácii a kroku 3"

        rows_out.append(
            {
                "Filter": f,
                "Exp (s)": e,
                "Status Calibration": cal_st,
                "Status Plate Solve": ps_st,
            }
        )

    return pd.DataFrame(rows_out)


def draft_runtime_status(
    db: VyvarDatabase,
    *,
    draft_id: int | None,
    archive_path: Path | str | None,
) -> dict[str, bool]:
    """Best-effort runtime status flags for quick UI decisions."""
    analyzed = False
    calibrated = False
    did = int(draft_id) if draft_id is not None else None
    if did is not None and did > 0:
        try:
            rows = db.fetch_draft_light_rows_for_quality(did)
            if rows:
                df = pd.DataFrame(rows)
                if "FWHM" in df.columns:
                    analyzed = bool(pd.to_numeric(df["FWHM"], errors="coerce").notna().any())
                if not analyzed and "INSPECTION_JD" in df.columns:
                    analyzed = bool(pd.to_numeric(df["INSPECTION_JD"], errors="coerce").notna().any())
        except Exception:  # noqa: BLE001
            analyzed = False

    if archive_path is not None:
        try:
            ap = Path(archive_path)
            ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
            for root in (ap_root / "calibrated" / "lights", ap_root / "non_calibrated" / "lights"):
                if not root.is_dir():
                    continue
                for pat in ("*.fits", "*.fit", "*.fts"):
                    if any(root.rglob(pat)):
                        calibrated = True
                        break
                if calibrated:
                    break
        except Exception:  # noqa: BLE001
            calibrated = False

    return {"analyzed": bool(analyzed), "calibrated": bool(calibrated)}
