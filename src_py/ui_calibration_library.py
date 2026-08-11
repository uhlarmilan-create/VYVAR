"""Streamlit dashboard for CalibrationLibrary validity overview, delete, and master generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from astropy.io import fits

from calibration import get_master_age_days, read_master_binning_from_header
from database import VyvarDatabase
from importer import generate_master_dark_from_source_dir, generate_master_flat_from_source_dir


def _iter_master_fits(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for pat in ("*.fits", "*.fit", "*.fts"):
        out.extend(list(root.rglob(pat)))
    return sorted(set(out))


def _parse_kind_from_name(path: Path, header: fits.Header) -> str | None:
    n = path.name.casefold()
    imgt = str(header.get("IMAGETYP") or "").strip().casefold()
    if "dark" in imgt or n.startswith("md_") or "dark" in n:
        return "dark"
    if "flat" in imgt or n.startswith("mf_") or "flat" in n:
        return "flat"
    return None


def _date_text(header: fits.Header, path: Path) -> str:
    raw = header.get("VY_CDATE") or header.get("DATE-OBS") or header.get("DATEOBS")
    if raw not in (None, ""):
        return str(raw)
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")


def _status_for_age(age_days: float, limit_days: int) -> str:
    lim = max(1.0, float(limit_days))
    if age_days > lim:
        return "Expired"
    if age_days >= 0.8 * lim:
        return "Warning"
    return "OK"


def _status_style(v: Any) -> str:
    t = str(v).strip().casefold()
    if t == "ok":
        return "background-color: #1f7a3f; color: #ffffff;"
    if t == "warning":
        return "background-color: #b26a00; color: #ffffff;"
    if t == "expired":
        return "background-color: #8b1e1e; color: #ffffff;"
    return ""


def _equipment_telescope_labels(tags: dict[str, Any] | None) -> tuple[str, str]:
    """Labels from :meth:`VyvarDatabase.calibration_library_path_tag_map` row, or placeholders."""
    if not tags:
        return "-", "-"
    ie, it = tags.get("id_equipments"), tags.get("id_telescope")
    if ie is None and it is None:
        return "General", "General"
    cam = tags.get("camera") or tags.get("eq_alias")
    tel = tags.get("telescope") or tags.get("tel_alias")
    return (
        str(cam) if cam else (f"Equipment #{ie}" if ie is not None else "-"),
        str(tel) if tel else (f"Telescope #{it}" if it is not None else "-"),
    )


def _build_rows(
    root: Path,
    *,
    dark_limit: int,
    flat_limit: int,
    tag_map: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dark_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for fp in _iter_master_fits(root):
        try:
            with fits.open(fp, memmap=True) as hdul:
                hdr = hdul[0].header
        except Exception:  # noqa: BLE001
            # EXC-0506: T3 -- UI diagnostic/plot only (with fits.open(fp, memmap=True) as hdul: / hdr = hdul[0].heade... (EXCEPT-BULK 2026-07-08)
            continue
        kind = _parse_kind_from_name(fp, hdr)
        if kind is None:
            continue
        age = float(get_master_age_days(fp))
        filt = str(hdr.get("FILTER") or hdr.get("FILT") or ("Dark" if kind == "dark" else "NoFilter"))
        expt = hdr.get("EXPTIME")
        try:
            expt_v = float(expt) if expt is not None else None
        except (TypeError, ValueError):
            expt_v = None
        bin_v = int(read_master_binning_from_header(hdr))
        date_s = _date_text(hdr, fp)
        path_res = str(fp.resolve())
        tkey = str(Path(path_res).resolve()).casefold()
        row_tags = tag_map.get(tkey) if tag_map else None
        cam_lbl, tel_lbl = _equipment_telescope_labels(row_tags)
        if kind == "dark":
            dark_rows.append(
                {
                    "Filter": filt,
                    "Exp (s)": expt_v,
                    "Bin": bin_v,
                    "Camera": cam_lbl,
                    "Telescope": tel_lbl,
                    "Date": date_s,
                    "Age (days)": round(age, 2),
                    "Status": _status_for_age(age, int(dark_limit)),
                    "File": fp.name,
                    "_path": path_res,
                }
            )
        else:
            flat_rows.append(
                {
                    "Filter": filt,
                    "Bin": bin_v,
                    "Camera": cam_lbl,
                    "Telescope": tel_lbl,
                    "Date": date_s,
                    "Age (days)": round(age, 2),
                    "Status": _status_for_age(age, int(flat_limit)),
                    "File": fp.name,
                    "_path": path_res,
                }
            )
    return dark_rows, flat_rows


_CLIB_DEL_PENDING_KEY = "clib_del_pending"
_CLIB_DEL_DIALOG_KEY = "clib_del_dialog_open"


def _delete_paths(paths: list[Path], db: VyvarDatabase | None) -> tuple[int, list[str]]:
    errors: list[str] = []
    n_ok = 0
    for p in paths:
        try:
            if p.is_file():
                p.unlink()
                n_ok += 1
            if db is not None:
                db.delete_calibration_library_entry_by_path(p)
        except OSError as exc:
            errors.append(f"{p.name}: {exc}")
    return n_ok, errors


def _make_delete_confirm_dialog(db: VyvarDatabase | None):
    """OK/CANCEL confirm for a single master file delete."""

    def _body() -> None:
        pending = st.session_state.get(_CLIB_DEL_PENDING_KEY)
        if not pending:
            return
        st.markdown(f"Delete **{pending['kind']}** master file?")
        st.text(str(pending.get("file") or ""))
        c_ok, c_cancel = st.columns(2)
        with c_ok:
            if st.button("OK", key="clib_del_confirm_ok", type="primary"):
                n_ok, errs = _delete_paths([Path(str(pending["path"]))], db)
                st.session_state.pop(_CLIB_DEL_PENDING_KEY, None)
                if errs:
                    st.error("Could not delete file:\n" + "\n".join(errs))
                elif n_ok:
                    st.success(f"Deleted: {pending.get('file')}")
                st.rerun()
        with c_cancel:
            if st.button("CANCEL", key="clib_del_confirm_cancel"):
                st.session_state.pop(_CLIB_DEL_PENDING_KEY, None)
                st.rerun()

    if hasattr(st, "dialog"):
        return st.dialog("Confirm delete")(_body)
    return _body


def _render_master_table_with_delete(
    rows: list[dict[str, Any]],
    *,
    kind_label: str,
    key_prefix: str,
    include_exptime: bool,
) -> None:
    """Overview table with a per-row Delete button (last column)."""
    sorted_rows = sorted(
        rows,
        key=lambda r: (str(r.get("Status") or ""), -float(r.get("Age (days)") or 0.0)),
    )
    if include_exptime:
        headers = [
            "Filter",
            "Exp (s)",
            "Bin",
            "Camera",
            "Telescope",
            "Date",
            "Age (days)",
            "Status",
            "File",
            "Delete",
        ]
        weights = [1, 1, 1, 2, 2, 2, 1, 1, 3, 1]
    else:
        headers = ["Filter", "Bin", "Camera", "Telescope", "Date", "Age (days)", "Status", "File", "Delete"]
        weights = [1, 1, 2, 2, 2, 1, 1, 3, 1]

    hdr_cols = st.columns(weights)
    for hdr, col in zip(headers, hdr_cols):
        col.markdown(f"**{hdr}**")

    for idx, row in enumerate(sorted_rows):
        cols = st.columns(weights)
        ci = 0
        cols[ci].write(row.get("Filter", ""))
        ci += 1
        if include_exptime:
            cols[ci].write(row.get("Exp (s)", ""))
            ci += 1
        cols[ci].write(row.get("Bin", ""))
        ci += 1
        cols[ci].write(row.get("Camera", ""))
        ci += 1
        cols[ci].write(row.get("Telescope", ""))
        ci += 1
        cols[ci].write(row.get("Date", ""))
        ci += 1
        cols[ci].write(row.get("Age (days)", ""))
        ci += 1
        status = row.get("Status", "")
        cols[ci].markdown(
            f'<span style="padding:2px 6px; border-radius:4px; {_status_style(status)}">{status}</span>',
            unsafe_allow_html=True,
        )
        ci += 1
        cols[ci].write(row.get("File", ""))
        ci += 1
        if cols[ci].button("Delete", key=f"{key_prefix}_del_{idx}", type="secondary"):
            st.session_state[_CLIB_DEL_PENDING_KEY] = {
                "path": row["_path"],
                "file": row.get("File"),
                "kind": kind_label,
            }
            st.session_state[_CLIB_DEL_DIALOG_KEY] = True


def render_calibration_library_dashboard(
    *,
    calibration_library_root: Path,
    dark_validity_days: int,
    flat_validity_days: int,
    db: VyvarDatabase | None = None,
) -> None:
    st.subheader("Calibration Library")
    st.caption("Overview of Master Dark/Flat frames, age, and validity.")
    st.caption(
        f"Limits: MasterDark = {int(dark_validity_days)} days, MasterFlat = {int(flat_validity_days)} days."
    )
    root = Path(calibration_library_root)
    st.caption(f"Library: `{root}`")
    if not root.is_dir():
        st.warning("CalibrationLibrary path does not exist.")
        return

    tag_map: dict[str, dict[str, Any]] | None = None
    if db is not None:
        try:
            tag_map = db.calibration_library_path_tag_map()
        except Exception:  # noqa: BLE001
            tag_map = None

    dark_rows, flat_rows = _build_rows(
        root,
        dark_limit=int(dark_validity_days),
        flat_limit=int(flat_validity_days),
        tag_map=tag_map,
    )

    delete_confirm_dialog = _make_delete_confirm_dialog(db)
    if st.session_state.pop(_CLIB_DEL_DIALOG_KEY, False):
        delete_confirm_dialog()

    st.markdown("**Master Darks**")
    if dark_rows:
        _render_master_table_with_delete(
            dark_rows,
            kind_label="Master Dark",
            key_prefix="clib_dark",
            include_exptime=True,
        )
    else:
        st.info("No Master Dark found in CalibrationLibrary.")

    st.markdown("**Master Flats**")
    if flat_rows:
        _render_master_table_with_delete(
            flat_rows,
            kind_label="Master Flat",
            key_prefix="clib_flat",
            include_exptime=False,
        )
    else:
        st.info("No Master Flat found in CalibrationLibrary.")

    st.markdown("---")
    st.markdown("**Generate masters into library**")
    st.caption(
        "Enter a directory with **raw** dark or flat FITS (including subfolders). "
        "Stacking and file naming (e.g. `Dark_120s_Dark_0G_-10deg_Bin2_YYYYMMDD.fits`, `Flat_...NoFilter...`) "
        "matches import - from headers EXPTIME, FILTER, GAIN, CCD temperature, binning, DATE-OBS."
    )
    st.caption(
        "**Required set:** before generating, select **camera (Equipment)** and **telescope (Telescope)** - "
        "the master is registered in the library under this set."
    )
    gen_eq_id: int | None = None
    gen_tel_id: int | None = None
    gen_set_ok = False
    if db is None:
        st.warning("Database unavailable - generation with library set registration is not possible.")
    else:
        gen_equipments = db.get_equipments(active_only=True)
        gen_telescopes = db.get_telescopes(active_only=True)
        gen_eq_opts = {
            f"{item['ID']}: {item['CAMERANAME']} ({item['ALIAS']})": int(item["ID"])
            for item in gen_equipments
        }
        gen_tel_opts = {
            f"{item['ID']}: {item['TELESCOPENAME']} ({item['ALIAS']})": int(item["ID"])
            for item in gen_telescopes
        }
        gel, gtl = list(gen_eq_opts.keys()), list(gen_tel_opts.keys())
        if not gel or not gtl:
            st.error(
                "The database must have at least one **active camera** and one **active telescope**. "
                "Without that, a set cannot be chosen for master registration."
            )
        else:
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                glab_eq = st.selectbox(
                    "Equipment (required)",
                    options=gel,
                    key="clib_gen_equipment",
                )
            with gcol2:
                glab_tel = st.selectbox(
                    "Telescope (required)",
                    options=gtl,
                    key="clib_gen_telescope",
                )
            gen_eq_id = int(gen_eq_opts[glab_eq])
            gen_tel_id = int(gen_tel_opts[glab_tel])
            gen_set_ok = True
    dark_src = st.text_input(
        "Path to raw dark frames",
        value="",
        key="clib_gen_dark_src",
        help="Directory containing dark FITS (IMAGETYP or filename).",
    )
    flat_src = st.text_input(
        "Path to raw flat frames",
        value="",
        key="clib_gen_flat_src",
        help="Directory with flat FITS.",
    )
    gc1, gc2 = st.columns(2)
    with gc1:
        if st.button(
            "Generate Master Dark",
            type="primary",
            key="clib_btn_gen_dark",
            disabled=not gen_set_ok,
        ):
            p = Path(dark_src.strip())
            if not gen_set_ok or gen_eq_id is None or gen_tel_id is None:
                st.error("Select camera and telescope (set).")
            elif not str(dark_src).strip():
                st.warning("Enter path to darks.")
            else:
                try:
                    out, msgs = generate_master_dark_from_source_dir(
                        source_dir=p,
                        calibration_library_root=root,
                        db=db,
                        id_equipments=gen_eq_id,
                        id_telescope=gen_tel_id,
                    )
                    for m in msgs:
                        if m.startswith("[OK]"):
                            st.success(m)
                        elif m.startswith("i"):
                            st.info(m)
                        else:
                            st.error(m)
                    if out is not None and all(not str(x).startswith("[X]") for x in msgs):
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error: {exc}")
    with gc2:
        if st.button(
            "Generate Master Flat",
            type="primary",
            key="clib_btn_gen_flat",
            disabled=not gen_set_ok,
        ):
            p = Path(flat_src.strip())
            if not gen_set_ok or gen_eq_id is None or gen_tel_id is None:
                st.error("Select camera and telescope (set).")
            elif not str(flat_src).strip():
                st.warning("Enter path to flats.")
            else:
                try:
                    out, msgs = generate_master_flat_from_source_dir(
                        source_dir=p,
                        calibration_library_root=root,
                        db=db,
                        id_equipments=gen_eq_id,
                        id_telescope=gen_tel_id,
                    )
                    for m in msgs:
                        if m.startswith("[OK]"):
                            st.success(m)
                        elif m.startswith("i"):
                            st.info(m)
                        else:
                            st.error(m)
                    if out is not None and all(not str(x).startswith("[X]") for x in msgs):
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error: {exc}")
