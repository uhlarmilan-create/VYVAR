"""Streamlit Calibration Library (CAL-2026-10, dev/results/specs/VYVAR_CALIBRATION_SPEC.md).

Shows every master with its camera / telescope link, origin (VYVAR or imported +
software), combine method, capture date and validity; builds masters from a folder
(median default, or mean - no clipping) and imports external masters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import calib_library as cl
import calib_masters as cm
import calib_plan as cp
from database import VyvarDatabase
from importer import _collect_fits_by_type, _looks_like_master

_KIND_LABEL = {cl.KIND_DARK: "DARK", cl.KIND_DARKFLAT: "DARKFLAT", cl.KIND_FLAT: "FLAT"}
_OFFSET_LABEL = {
    cl.OFFSET_DARKFLAT: "dark-flat removed",
    cl.OFFSET_CONSTANT: "constant offset removed",
    cl.OFFSET_NONE: "not removed",
    cl.OFFSET_ALREADY: "removed (by other software)",
    cl.OFFSET_UNKNOWN: "removed at use",
    "": "",
}
_DEL_PENDING = "clib_del_pending"


def _validity_days(kind: str, dark_days: int, flat_days: int) -> int:
    return int(flat_days if kind == cl.KIND_FLAT else dark_days)


def _validity(m: cl.LibraryMaster, days: int) -> tuple[str, int | None]:
    """("valid" / "expired" / "no date", days still valid - negative = days since it expired)."""
    vu = m.valid_until(days)
    if vu is None:
        return "no date", None
    left = (vu - datetime.now(timezone.utc)).days
    return ("expired" if left < 0 else "valid"), left


def _rows(masters: list[cl.LibraryMaster], dark_days: int, flat_days: int) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    for m in masters:
        days = _validity_days(m.kind, dark_days, flat_days)
        vu = m.valid_until(days)
        status, left = _validity(m, days)
        recs.append(
            {
                "Camera": m.camera_name or (f"#{m.camera_id}" if m.camera_id else "-"),
                "Telescope": (m.telescope_name or (f"#{m.telescope_id}" if m.telescope_id else "-"))
                if m.kind == cl.KIND_FLAT else (m.telescope_name or "-"),
                "Kind": _KIND_LABEL.get(m.kind, m.kind),
                "Bin": m.binning,
                "Exp (s)": m.exptime,
                "Filter": m.filter_name if m.kind == cl.KIND_FLAT else "",
                "Gain": m.gain,
                "Offset": m.offset,
                "Temp (C)": m.temp,
                "Captured": m.capture_utc.strftime("%Y-%m-%d") if m.capture_utc else "-",
                "Valid until": vu.strftime("%Y-%m-%d") if vu else "-",
                "Status": status,
                "Days left": left,
                "Made by": (m.origin_software or "VYVAR") if m.origin == cl.ORIGIN_IMPORTED else "VYVAR",
                "Origin": "imported" if m.origin == cl.ORIGIN_IMPORTED else "built",
                "Method": m.combine_method or "?",
                "Flat offset": _OFFSET_LABEL.get(m.offset_handling, m.offset_handling) if m.kind == cl.KIND_FLAT else "",
                "Frames": m.ncombine,
                "File": Path(m.path).name,
                "_path": m.path,
            }
        )
    df = pd.DataFrame(recs)
    if "Days left" in df.columns:
        df["Days left"] = df["Days left"].astype("Int64")
    return df


def _camera_options(db: VyvarDatabase) -> dict[str, int]:
    return {f"{e['ID']}: {e['CAMERANAME']} ({e['ALIAS']})": int(e["ID"]) for e in db.get_equipments(active_only=True)}


def _telescope_options(db: VyvarDatabase) -> dict[str, int]:
    return {f"{t['ID']}: {t['TELESCOPENAME']} ({t['ALIAS']})": int(t["ID"]) for t in db.get_telescopes(active_only=True)}


def _show_msgs(msgs: list[str]) -> None:
    for m in msgs:
        if m.startswith("[OK]"):
            st.success(m)
        elif m.startswith("[X]"):
            st.error(m)
        elif m.startswith("!"):
            st.warning(m)
        else:
            st.info(m)


def _render_table(db: VyvarDatabase, root: Path, dark_days: int, flat_days: int) -> None:
    cams = _camera_options(db)
    pick = st.selectbox("Camera", ["All cameras", *cams.keys()], key="clib_cam_filter")
    cam_id = cams.get(pick)
    masters = cl.list_masters(db, camera_id=cam_id)
    if not masters:
        st.info("No master registered for this selection.")
    else:
        df = _rows(masters, dark_days, flat_days)
        _color = {"valid": "color: #3dd68c", "expired": "color: #ff6b6b"}
        st.dataframe(
            df.drop(columns=["_path"]).style.map(lambda v: _color.get(v, ""), subset=["Status"]),
            hide_index=True,
            width="stretch",
            column_config={
                "Days left": st.column_config.NumberColumn(
                    "Days left", format="%d", help="Days the master is still valid; negative = days since it expired."
                ),
            },
        )
        files = list(df["File"])
        c1, c2 = st.columns([3, 1])
        with c1:
            sel = st.selectbox("Master to delete", ["-", *files], key="clib_del_pick")
        with c2:
            st.write("")
            if st.button("Delete", key="clib_del_btn", disabled=sel == "-"):
                st.session_state[_DEL_PENDING] = str(df.loc[df["File"] == sel, "_path"].iloc[0])
        pending = st.session_state.get(_DEL_PENDING)
        if pending:
            st.warning(f"Delete {Path(pending).name} from disk and library?")
            o1, o2 = st.columns(2)
            if o1.button("Yes, delete", key="clib_del_yes", type="primary"):
                try:
                    p = Path(pending)
                    if p.is_file():
                        p.unlink()
                    db.delete_calibration_library_entry_by_path(p)
                    st.success(f"Deleted {p.name}")
                except OSError as exc:
                    st.error(f"Could not delete: {exc}")
                st.session_state.pop(_DEL_PENDING, None)
                st.rerun()
            if o2.button("Cancel", key="clib_del_no"):
                st.session_state.pop(_DEL_PENDING, None)
                st.rerun()
    registered = {str(Path(m.path).resolve()).casefold() for m in cl.list_masters(db)}
    loose = [
        p for pat in ("*.fits", "*.fit", "*.fts") for p in root.rglob(pat)
        if str(p.resolve()).casefold() not in registered
    ]
    if loose:
        with st.expander(f"{len(loose)} file(s) in the library folder are not registered (not used)"):
            st.caption("Use 'Import a master' below to link them to a camera.")
            st.write(sorted(p.name for p in loose))


def _render_build(db: VyvarDatabase, root: Path) -> None:
    st.markdown("**Build masters from raw frames (DARK, DARKFLAT, FLAT)**")
    st.caption(
        "Folder with raw frames (subfolders included, recognised by IMAGETYP). Auto: darks with the "
        "exposure of the flats become DARKFLAT. Flats get their offset removed (DARKFLAT from the "
        "library, else a constant offset measured from darks). No clipping."
    )
    cams, tels = _camera_options(db), _telescope_options(db)
    if not cams or not tels:
        st.error("Add at least one active camera and one telescope in the database first.")
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    cam = cams[c1.selectbox("Camera", list(cams), key="clib_b_cam")]
    tel = tels[c2.selectbox("Telescope (for flats)", list(tels), key="clib_b_tel")]
    method = c3.radio("Combine", ["median", "mean"], index=0, key="clib_b_method",
                      help="Median (default) rejects cosmic rays; mean has lower noise. No clipping either way.")
    what = st.radio("Build", ["Auto (by IMAGETYP / exposure)", "DARK", "DARKFLAT", "FLAT"], index=0,
                    horizontal=True, key="clib_b_what")
    src = st.text_input("Folder with raw calibration frames", key="clib_b_src")
    if st.button("Build masters", type="primary", key="clib_b_go", disabled=not src.strip()):
        p = Path(src.strip())
        if not p.is_dir():
            st.error(f"Folder not found: {p}")
            return
        by = _collect_fits_by_type(p, db=db)
        darks = [f for f in by.get("dark", []) if not _looks_like_master(f)]
        flats = [f for f in by.get("flat", []) if not _looks_like_master(f)]
        dark_kind = None
        if what in ("DARK", "DARKFLAT"):
            dark_kind, flats = what.lower(), []
            # dark-flat frames may be typed FLAT DARK; they are dark frames here
            darks = darks + [f for f in by.get("flat", []) if not _looks_like_master(f) and _is_dark_typed(f)]
        elif what == "FLAT":
            darks = []
            flats = [f for f in flats if not _is_dark_typed(f)]
        if not darks and not flats:
            st.error(f"No raw {what if what != 'Auto (by IMAGETYP / exposure)' else 'DARK or FLAT'} "
                     "frames found (IMAGETYP must say dark / flat).")
            return
        with st.spinner(f"Building masters from {len(darks)} dark and {len(flats)} flat frames..."):
            try:
                bayer = db.get_equipment_bayermask(int(cam))
            except Exception:  # noqa: BLE001
                bayer = None
            built, msgs = cp.build_masters_from_source(
                dark_files=darks, flat_files=flats, light_exposures=[], db=db, camera_id=cam,
                telescope_id=tel, library_root=root, method=method, equipment_bayer_mask=bayer,
                dark_kind=dark_kind,
            )
        _show_msgs(msgs)
        if built:
            st.rerun()


def _is_dark_typed(fp: Path) -> bool:
    try:
        from astropy.io import fits  # noqa: PLC0415

        with fits.open(fp, memmap=True) as h:
            return cl.header_kind(h[0].header) in (cl.KIND_DARK, cl.KIND_DARKFLAT)
    except (OSError, ValueError):
        return False


def _render_import(db: VyvarDatabase, root: Path) -> None:
    st.markdown("**Import a master made by another program**")
    st.caption("The file is copied into the library and linked to the camera (and telescope for a FLAT).")
    cams, tels = _camera_options(db), _telescope_options(db)
    if not cams or not tels:
        return
    path = st.text_input("Master FITS file, or a folder with master files", key="clib_i_path")
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Kind", ["Auto (IMAGETYP)", "DARK", "DARKFLAT", "FLAT"], key="clib_i_kind").lower()
    if kind.startswith("auto"):
        kind = "auto"
    cam = cams[c2.selectbox("Camera", list(cams), key="clib_i_cam")]
    tel = tels[c3.selectbox("Telescope", list(tels), key="clib_i_tel")]
    sw = st.text_input("Made with (software)", key="clib_i_sw",
                       help="e.g. SIPS, AstroImageJ, PixInsight. Empty = read from the FITS header.")
    off_removed = None
    if kind in (cl.KIND_FLAT, "auto"):
        _ans = st.radio(
            "Was the offset (dark-flat / offset level) already removed from this flat?",
            ["Read from the header", "Yes", "No / not sure"], index=0, key="clib_i_off",
            help="Header: VYVAR stamp or HISTORY 'Dark frame subtracted' (C-Munipack) means removed. "
                 "If not removed, VYVAR measures the offset from your darks and removes it when the flat is used.",
        )
        off_removed = None if _ans.startswith("Read") else (_ans == "Yes")
    if st.button("Import", type="primary", key="clib_i_go", disabled=not path.strip()):
        src = Path(path.strip())
        files = (
            sorted(f for pat in ("*.fits", "*.fit", "*.fts") for f in src.rglob(pat) if _looks_like_master(f))
            if src.is_dir() else [src]
        )
        if not files:
            st.error(f"No master file found in {src}")
            return
        n_ok = 0
        for f in files:
            k = kind
            if k == "auto":
                try:
                    from astropy.io import fits  # noqa: PLC0415

                    with fits.open(f, memmap=True) as h:
                        k = cl.header_kind(h[0].header) or ""
                except (OSError, ValueError):
                    k = ""
                if k not in cl.KINDS:
                    st.warning(f"{f.name}: IMAGETYP does not say dark / flat - choose the kind")
                    continue
            r = cm.import_master(f, kind=k, library_root=root, db=db, camera_id=cam,
                                 telescope_id=tel if k == cl.KIND_FLAT else None,
                                 origin_software=sw.strip(),
                                 flat_offset_removed=off_removed if k == cl.KIND_FLAT else None)
            _show_msgs(r.messages)
            n_ok += int(r.master is not None)
        if n_ok:
            st.success(f"{n_ok} master(s) in the library")


def _render_measure_gain(db: VyvarDatabase) -> None:
    """CAMERA-PROFILE-2026-10: gain and read noise of a camera from raw flat and dark pairs."""
    import datetime as _dt  # noqa: PLC0415

    from camera_profile import profile_from_connection  # noqa: PLC0415
    from gain_ptc import measure_from_files, native_gain  # noqa: PLC0415

    st.markdown("**Measure gain and read noise from raw flats (photon transfer)**")
    st.caption(
        "Folder with RAW (not mastered) frames of one gain setting and binning: flats in pairs - two "
        "consecutive flats of the same exposure, best at 3-5 exposures from about 10 % to 60 % of "
        "saturation - and two darks of each flat exposure (the shortest pair also gives the read noise). "
        "Masters cannot be used: averaging removes the noise this method measures."
    )
    cams = _camera_options(db)
    if not cams:
        return
    c1, c2 = st.columns([2, 3])
    cam = cams[c1.selectbox("Camera", list(cams), key="clib_g_cam")]
    src = c2.text_input("Folder with raw flats and darks", key="clib_g_src")
    if st.button("Measure", key="clib_g_go", disabled=not src.strip()):
        p = Path(src.strip())
        if not p.is_dir():
            st.error(f"Folder not found: {p}")
            return
        by = _collect_fits_by_type(p, db=db)
        flats = [f for f in by.get("flat", []) if not _looks_like_master(f) and not _is_dark_typed(f)]
        darks = [f for f in by.get("dark", []) if not _looks_like_master(f)]
        prof = profile_from_connection(db.conn, int(cam))
        sat = db.conn.execute("SELECT SATURATE_ADU FROM EQUIPMENTS WHERE ID = ?;", (int(cam),)).fetchone()
        sat_adu = float(sat[0]) if sat and sat[0] else 65535.0
        with st.spinner(f"Measuring {len(flats)} flats and {len(darks)} darks..."):
            res = measure_from_files(flats, darks, saturation_adu=sat_adu)
        st.session_state["clib_g_res"] = {"cam": int(cam), "res": res, "adc": prof.adc_bits, "mode": prof.mode_key}
    out = st.session_state.get("clib_g_res")
    if not out or out.get("cam") != int(cam):
        return
    res = out["res"]
    for m in res.messages:
        st.warning(m)
    if res.points:
        pts = pd.DataFrame([
            {"Flats": " + ".join(pt.files), "Exposure s": pt.exptime, "Level ADU": round(pt.level_adu, 1),
             "var/2 ADU^2": round(pt.var_half, 2), "Used": pt.used, "Note": pt.note} for pt in res.points
        ])
        st.dataframe(pts, hide_index=True, width="stretch")
        st.scatter_chart(pts[pts["Used"]], x="Level ADU", y="var/2 ADU^2")
    if not res.ok:
        return
    g_nat = native_gain(float(res.gain), out["adc"])
    err = f" +- {native_gain(float(res.gain_err), out['adc']):.3f}" if res.gain_err else ""
    rn_txt = f"{res.read_noise_e:.2f} e-" if res.read_noise_e else "not measured (no dark pair)"
    st.success(
        f"Gain {g_nat:.3f}{err} e-/ADU native ({res.gain:.4f} on the file values), read noise {rn_txt}, "
        f"{res.n_levels} level(s), R^2 = {res.r2 if res.r2 is not None else float('nan'):.5f}; "
        f"gain setting {res.gain_setting if res.gain_setting is not None else '-'}, bin {res.binning}."
    )
    if st.button("Save to camera", type="primary", key="clib_g_save"):
        db.conn.execute(
            "INSERT INTO EQUIPMENT_CURVE (EQUIPMENT_ID, READOUT_MODE, GAIN_SETTING, BINNING, E_PER_ADU, "
            "READNOISE_E, SOURCE, MEASURED_AT, NOTE) VALUES (?, ?, ?, ?, ?, ?, 'measured', ?, ?);",
            (
                int(out["cam"]), out["mode"], res.gain_setting, int(res.binning), g_nat, res.read_noise_e,
                _dt.date.today().isoformat(),
                f"photon transfer, {res.n_levels} levels, R2 {res.r2 if res.r2 is not None else float('nan'):.4f}",
            ),
        )
        db.conn.commit()
        st.session_state.pop("clib_g_res", None)
        st.success("Saved. It now wins over the catalog and the camera default for this setting and binning.")


def render_calibration_library_dashboard(
    *,
    calibration_library_root: Path,
    dark_validity_days: int,
    flat_validity_days: int,
    db: VyvarDatabase | None = None,
) -> None:
    st.subheader("Calibration Library")
    st.caption(
        f"Validity: DARK / DARKFLAT {int(dark_validity_days)} d, FLAT {int(flat_validity_days)} d. "
        "A master is used for LIGHTs taken from 1 day before its capture date until the end of its validity."
    )
    root = Path(calibration_library_root)
    st.caption(f"Library folder: `{root}`")
    if db is None:
        st.warning("Database unavailable.")
        return
    root.mkdir(parents=True, exist_ok=True)
    cl.ensure_schema(db)
    _render_table(db, root, int(dark_validity_days), int(flat_validity_days))
    st.markdown("---")
    _render_build(db, root)
    st.markdown("---")
    _render_import(db, root)
    st.markdown("---")
    _render_measure_gain(db)
