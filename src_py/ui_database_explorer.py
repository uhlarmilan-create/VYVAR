"""Database Explorer tab: cameras (with gain curve and catalog), telescopes and observing sites -
add, edit, set default, hide and delete (REF-OBJECTS-2026-10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from database import VyvarDatabase
from pipeline import AstroPipeline


def _read_df(conn, sql, params=()):
    """Read a query into a DataFrame via the RLock-serialized wrapper.

    Avoids pandas' DBAPI cursor() path, which ThreadSafeSQLiteConnection does not expose.
    """
    cur = conn.execute(sql, tuple(params))
    try:
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchall()  # _LockedCursor.fetchall releases the RLock in finally
    except Exception:
        cur.close()  # release the RLock if we failed before fetchall
        raise
    return pd.DataFrame.from_records(rows, columns=cols)


def equipment_detector_warnings(rows: list[dict[str, Any]]) -> list[str]:
    """Cameras whose gain / read noise is unknown at their default setting or implausible (pure; unit
    tested). ``rows``: ``ID, CAMERANAME, ALIAS`` plus the resolved ``GAIN`` / ``RN`` (None = unknown)."""
    out: list[str] = []
    for r in rows:
        name = str(r.get("CAMERANAME") or r.get("ALIAS") or f"ID={r.get('ID')}").strip()

        def num(v: Any) -> float | None:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f if f == f else None

        g, rn = num(r.get("GAIN")), num(r.get("RN"))
        if g is None or g <= 0 or rn is None or rn <= 0:
            out.append(
                f"{name}: gain or read noise unknown at its default setting - photometry stops for frames "
                "of this camera. Choose the camera model, or enter GAIN_ADU / READNOISE_E for the setting you use."
            )
            continue
        if g > 50:
            out.append(f"{name}: gain {g:g} e-/ADU is unusually high - check the units.")
        if rn > 50:
            out.append(f"{name}: read noise {rn:g} e- is unusually high (typical 1-15 e-) - check the units.")
    return out


def _active_profile_rows(pipeline: AstroPipeline) -> list[dict[str, Any]]:
    from camera_profile import lookup, profile_from_connection  # noqa: PLC0415

    df = _read_df(
        pipeline.db.conn,
        "SELECT ID, CAMERANAME, ALIAS FROM EQUIPMENTS "
        f"WHERE {VyvarDatabase.sql_expr_active_is_true('ACTIVE')} ORDER BY ID;",
    )
    rows = []
    for r in df.to_dict("records"):
        prof = profile_from_connection(pipeline.db.conn, int(r["ID"]))
        r["GAIN"] = lookup(prof, "gain", None, 1).value
        r["RN"] = lookup(prof, "read_noise", None, 1).value
        rows.append(r)
    return rows


def _render_equipment_detector_check(pipeline: AstroPipeline) -> None:
    warns = equipment_detector_warnings(_active_profile_rows(pipeline))
    if warns:
        st.warning("Detector values to check (active cameras):\n\n" + "\n".join(f"- {w}" for w in warns))
    else:
        st.caption("Gain and read noise are known at the default setting of every active camera.")


_SOURCE_BADGE = {"measured": "measured", "manual": "manual", "equipment": "camera default", "catalog": "catalog"}


def _model_label(m: dict[str, Any]) -> str:
    return f"{m.get('vendor')} {m.get('model')} - {m.get('sensor')} ({m.get('color')})"


def _render_cameras(pipeline: AstroPipeline) -> None:
    """CAMERA-PROFILE-2026-10: cameras, their gain curve and the camera catalog."""
    from camera_profile import (  # noqa: PLC0415
        catalog_missing_message,
        catalog_model,
        catalog_models,
        lookup,
        profile_from_connection,
    )

    t_cams, t_curve, t_cat = st.tabs(["List & edit", "Gain curve", "Camera catalog"])
    models = catalog_models()
    _missing = catalog_missing_message()
    if _missing:
        st.error(_missing)
    with t_cams:
        _render_objects(pipeline, "EQUIPMENTS")
        _render_equipment_detector_check(pipeline)

    cams = _read_df(pipeline.db.conn, "SELECT ID, CAMERANAME FROM EQUIPMENTS ORDER BY ID;")
    with t_curve:
        if cams.empty:
            st.info("No camera yet - add one in the Cameras tab.")
        else:
            labels = {f"{n} (ID {int(i)})": int(i) for i, n in zip(cams["ID"], cams["CAMERANAME"].astype(str))}
            eid = labels[st.selectbox("Camera", list(labels), key="vyvar_curve_cam")]
            prof = profile_from_connection(pipeline.db.conn, int(eid))
            model = catalog_model(prof.model_key)
            st.caption(
                f"Model: {_model_label(model) if model else 'not chosen'} | readout mode: {prof.mode_key or '-'} | "
                f"binning convention: {prof.bin_mode} ({prof.bin_mode_source}) | header GAIN card: "
                f"{prof.header_gain or 'decided from its comment'} | default setting: "
                f"{'any' if prof.default_setting is None else f'{prof.default_setting:g}'} | ADC bits: {prof.adc_bits or '-'}"
            )
            st.markdown("**Your points** (measured or entered; they win over the catalog)")
            df = _read_df(
                pipeline.db.conn,
                "SELECT ID, EQUIPMENT_ID, READOUT_MODE, GAIN_SETTING, BINNING, E_PER_ADU, READNOISE_E, SOURCE, "
                "MEASURED_AT, NOTE FROM EQUIPMENT_CURVE WHERE EQUIPMENT_ID = ? ORDER BY GAIN_SETTING, BINNING;",
                (int(eid),),
            )
            edited = st.data_editor(
                df,
                width="stretch",
                num_rows="dynamic",
                hide_index=True,
                disabled=["ID", "EQUIPMENT_ID"],
                column_config={
                    "SOURCE": st.column_config.SelectboxColumn("SOURCE", options=["manual", "measured"], required=True),
                    "BINNING": st.column_config.NumberColumn("BINNING", min_value=1, max_value=16, step=1),
                },
                key=f"vyvar_curve_ed_{eid}",
            )
            if st.button("Save points", key=f"vyvar_curve_save_{eid}"):
                try:
                    n = pipeline.db.save_equipment_curve(int(eid), edited)
                    st.success(f"Saved {n} point(s).")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
            cat = [pt for pt in prof.points if pt.source == "catalog"]
            if cat:
                st.markdown("**Catalog points** (typical values, read-only)")
                st.dataframe(
                    pd.DataFrame([
                        {"GAIN_SETTING": pt.setting, "E_PER_ADU": pt.e_per_adu, "READNOISE_E": pt.read_noise_e,
                         "SOURCE": pt.note} for pt in cat
                    ]),
                    hide_index=True, width="stretch",
                )
            st.markdown("**Check a frame**")
            c1, c2 = st.columns(2)
            s_txt = c1.text_input("Header gain setting (empty = none in the header)", value="", key=f"vyvar_curve_s_{eid}")
            b_val = int(c2.number_input("Binning", min_value=1, max_value=16, value=1, step=1, key=f"vyvar_curve_b_{eid}"))
            try:
                s_val = float(s_txt) if s_txt.strip() else None
            except ValueError:
                s_val = None
                st.error("The gain setting must be a number.")
            g = lookup(prof, "gain", s_val, b_val)
            r = lookup(prof, "read_noise", s_val, b_val)
            from gain_ptc import container_scale  # noqa: PLC0415

            scale = container_scale(prof.adc_bits)
            if g.value is None or r.value is None:
                st.warning(f"Not known: {g.detail if g.value is None else r.detail}. Photometry would stop here.")
            else:
                st.success(
                    f"Gain {g.value:.4g} e-/ADU ({g.detail}); on the 16-bit "
                    f"pixel values {g.value / scale:.4g} e-/ADU. Read noise {r.value:.4g} e- "
                    f"({_SOURCE_BADGE.get(r.source, r.source)})."
                )

    with t_cat:
        rows = []
        for m in models:
            for md in m.get("modes", []):
                rows.append({
                    "Vendor": m.get("vendor"), "Model": m.get("model"), "Sensor": m.get("sensor"),
                    "Color": m.get("color"), "Mode": md.get("name"), "Pixel um": md.get("pixel_um"),
                    "ADC bits": md.get("adc_bits"), "Binning": md.get("binning") or "-",
                    "Header GAIN": m.get("header_gain"), "Curve points": len(md.get("curve") or []),
                    "Source": "; ".join(m.get("sources") or []),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Typical manufacturer values shipped with VYVAR. Your measured or entered points always win.")


@dataclass(frozen=True)
class _Field:
    col: str
    label: str
    kind: str = "float"  # text | float | int | select
    lo: float | None = None
    hi: float | None = None
    help: str = ""
    options: tuple[str, ...] = ()
    required: bool = False
    section: str = ""


_BAYER = ("", "RGGB", "BGGR", "GBRG", "GRBG")
_S_CAM, _S_GAIN, _S_OTHER = "Camera", "Gain and noise at the setting you use", "Other"

#: REF-OBJECTS-2026-10: what the user sees and edits for cameras, telescopes and sites.
#: "list": (column, header, relative width) of the table; every row ends with Edit / Delete.
_SPECS: dict[str, dict[str, Any]] = {
    "EQUIPMENTS": {
        "one": "camera", "name": "CAMERANAME",
        "list": [("CAMERANAME", "Name", 1.6), ("ALIAS", "Note", 1.6), ("MODEL_KEY", "Model (catalog)", 2.2),
                 ("SENSORTYPE", "Sensor", 1.1), ("PIXELSIZE", "Pixel um", 0.8), ("GAIN_ADU", "Gain e-/ADU", 0.9),
                 ("READNOISE_E", "Read noise e-", 0.9), ("DEFAULT_GAIN_SETTING", "Gain setting", 0.9),
                 ("ADC_BITS", "ADC", 0.6)],
        "fields": [
            _Field("CAMERANAME", "Name", "text", required=True, section=_S_CAM),
            _Field("ALIAS", "Note", "text", section=_S_CAM),
            _Field("MODEL_KEY", "Model (catalog, type to search)", "select", section=_S_CAM),
            _Field("READOUT_MODE", "Readout mode", "select", section=_S_CAM),
            _Field("SENSORTYPE", "Sensor", "text", section=_S_CAM),
            _Field("SENSORSIZE", "Sensor size (px, W*H)", "text", section=_S_CAM),
            _Field("PIXELSIZE", "Pixel size (um, native)", "float", 0.5, 50.0, section=_S_CAM),
            _Field("DEFAULT_GAIN_SETTING", "Gain setting", "float", 0.0, 1.0e5,
                   "Driver gain setting you normally use (e.g. 0, 100). Empty = any setting.", section=_S_GAIN),
            _Field("GAIN_ADU", "Gain (e-/ADU, bin 1)", "float", 0.001, 100.0, section=_S_GAIN),
            _Field("READNOISE_E", "Read noise (e-, bin 1)", "float", 0.01, 200.0, section=_S_GAIN),
            _Field("ADC_BITS", "ADC bits", "int", 8, 16, "14 for IMX492 / IMX533, 16 for IMX571 / IMX455 / IMX411",
                   section=_S_GAIN),
            _Field("BIN_MODE", "Binning convention", "select", options=("", "average", "sum", "charge"),
                   help="average: gain x N^2, read noise x N; sum: gain x 1, read noise x N; charge: both x 1. "
                        "Empty = from the catalog model, else average.", section=_S_GAIN),
            _Field("HEADER_GAIN", "Header GAIN card means", "select", options=("", "e_per_adu", "setting"),
                   help="Empty = from the catalog model, else decided from the card comment.", section=_S_GAIN),
            _Field("SATURATE_ADU", "Saturation (ADU)", "float", 1.0, 1.0e7, section=_S_OTHER),
            _Field("BAYERMASK", "Bayer mask (OSC)", "select", options=_BAYER, section=_S_OTHER),
            _Field("SIGMA_SYS_MAG", "Systematic floor (mag)", "float", 0.0, 1.0,
                   "Used when the night has too few comparison stars to measure it.", section=_S_OTHER),
        ],
    },
    "TELESCOPE": {
        "one": "telescope", "name": "TELESCOPENAME",
        "list": [("TELESCOPENAME", "Name", 2.2), ("ALIAS", "Note", 2.2), ("DIAMETER", "Aperture mm", 1.0),
                 ("FOCAL", "Focal length mm", 1.0)],
        "fields": [
            _Field("TELESCOPENAME", "Name", "text", required=True),
            _Field("ALIAS", "Note", "text"),
            _Field("DIAMETER", "Aperture (mm)", "float", 1.0, 20000.0,
                   "Needed for the scintillation term of the photometric errors (C3, Milan 2026-10-29).", required=True),
            _Field("FOCAL", "Focal length (mm)", "float", 1.0, 120000.0, "Effective focal length incl. reducer."),
        ],
    },
    "LOCATION": {
        "one": "site", "name": "PLACENAME",
        "list": [("PLACENAME", "Name", 3.0), ("LATITUDE", "Latitude", 1.0), ("LONGITUDE", "Longitude", 1.0),
                 ("ALTITUDE", "Altitude m", 1.0)],
        "fields": [
            _Field("PLACENAME", "Name", "text", required=True),
            _Field("LATITUDE", "Latitude (deg, north +)", "float", -90.0, 90.0),
            _Field("LONGITUDE", "Longitude (deg, east +)", "float", -180.0, 180.0),
            _Field("ALTITUDE", "Altitude (m)", "float", -500.0, 9000.0,
                   "Needed for the scintillation term of the photometric errors.", required=True),
        ],
    },
}


def _is_empty(v: Any) -> bool:
    if v is None or (isinstance(v, str) and not v.strip()):
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _fmt_cell(col: str, v: Any) -> str:
    if _is_empty(v):
        return "-"
    if col == "MODEL_KEY":
        from camera_profile import catalog_model  # noqa: PLC0415

        m = catalog_model(str(v))
        return f"{m.get('vendor')} {m.get('model')}" if m else str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _field_widget(f: _Field, row: dict[str, Any], key: str, container: Any) -> Any:
    cur = row.get(f.col)
    cur = None if _is_empty(cur) else cur
    if f.kind == "text":
        return container.text_input(f.label + (" *" if f.required else ""), value=str(cur or ""), key=key, help=f.help or None)
    if f.kind == "select":
        opts = list(f.options)
        val = "" if cur is None else str(cur)
        if val not in opts:
            opts.append(val)
        if f.col == "MODEL_KEY":
            from camera_profile import catalog_model  # noqa: PLC0415

            def _fmt(o: str) -> str:
                m = catalog_model(o) if o else None
                return _model_label(m) if m else (o or "(not set)")
        else:
            def _fmt(o: str) -> str:
                return o or "(not set)"
        return container.selectbox(f.label, opts, index=opts.index(val), key=key, help=f.help or None, format_func=_fmt)
    step = 1 if f.kind == "int" else None
    num = None if cur is None else (int(float(cur)) if f.kind == "int" else float(cur))
    return container.number_input(
        f.label + (" *" if f.required else ""), value=num, min_value=(int(f.lo) if f.kind == "int" else f.lo) if f.lo is not None else None,
        max_value=(int(f.hi) if f.kind == "int" else f.hi) if f.hi is not None else None,
        step=step, key=key, help=f.help or None, format=None if f.kind == "int" else "%g",
    )


def _row(df: pd.DataFrame, obj_id: int | None) -> dict[str, Any]:
    if obj_id is None or df.empty:
        return {}
    hit = df[df["ID"] == int(obj_id)]
    return {} if hit.empty else hit.iloc[0].to_dict()


def _load(pipeline: AstroPipeline, table: str) -> pd.DataFrame:
    df = _read_df(pipeline.db.conn, f"SELECT * FROM {table} ORDER BY ID;")
    if not df.empty:
        df["ACTIVE"] = df["ACTIVE"].map(VyvarDatabase.normalize_active_text)
        df["IS_DEFAULT"] = df["IS_DEFAULT"].map(VyvarDatabase.normalize_active_db_value).fillna(0).astype(int)
    return df


def _edit_body(pipeline: AstroPipeline, table: str, obj_id: int | None) -> None:
    """Content of the Edit / Add window."""
    from camera_profile import catalog_model, catalog_models  # noqa: PLC0415

    spec = _SPECS[table]
    one, name_col = spec["one"], spec["name"]
    db = pipeline.db
    df = _load(pipeline, table)
    row = _row(df, obj_id)
    kp = f"ref_{table}_{obj_id if obj_id is not None else 'new'}"
    if table == "EQUIPMENTS" and obj_id is None:
        how = st.radio("Start from", ["Camera catalog", "Empty form"], horizontal=True, key=f"{kp}_how")
        if how == "Camera catalog":
            _render_add_from_catalog(pipeline)
            return
    with st.form(key=f"{kp}_form", border=False):
        values: dict[str, Any] = {}
        sections = list(dict.fromkeys(f.section for f in spec["fields"]))
        for sec in sections:
            if sec:
                st.markdown(f"**{sec}**")
            fields = [f for f in spec["fields"] if f.section == sec]
            cols = st.columns(3)
            for i, f in enumerate(fields):
                fld = f
                if f.col == "MODEL_KEY":
                    fld = _Field(f.col, f.label, "select", options=("", *[m["key"] for m in catalog_models()]),
                                 help="A model from the catalog sets typical values (binning, header GAIN meaning, curve).")
                elif f.col == "READOUT_MODE":
                    m = catalog_model(row.get("MODEL_KEY") if not _is_empty(row.get("MODEL_KEY")) else None) or {}
                    fld = _Field(f.col, f.label, "select",
                                 options=("", *[str(md.get("key")) for md in m.get("modes", [])]),
                                 help="Readout mode of the model (save the model first to see its modes).")
                values[f.col] = _field_widget(fld, row, f"{kp}_{f.col}", cols[i % 3])
        b1, b2 = st.columns([1, 1])
        saved = b2.form_submit_button("Save changes" if obj_id is not None else f"Add {one}", type="primary",
                                      width="stretch")
        cancel = b1.form_submit_button("Cancel", width="stretch")
    if cancel:
        st.rerun()
    if saved:
        missing = [f.label for f in spec["fields"] if f.required and _is_empty(values.get(f.col))]
        if missing:
            st.error("Required: " + ", ".join(missing))
            return
        try:
            clean = {k: (None if _is_empty(v) else v) for k, v in values.items()}
            db.save_reference_object(table, obj_id, clean)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
    if obj_id is not None:
        is_default = int(row.get("IS_DEFAULT") or 0) == 1
        active = row.get("ACTIVE") == "YES"
        c1, c2 = st.columns(2)
        if c1.button("Set as default", key=f"{kp}_def", disabled=is_default, width="stretch",
                     help=f"The default {one} is pre-selected when you import a night."):
            db.set_reference_default(table, obj_id)
            st.rerun()
        if c2.button("Hide" if active else "Show again", key=f"{kp}_hide", disabled=is_default and active,
                     width="stretch", help=f"A hidden {one} is not offered at import; its data stay."):
            try:
                db.set_reference_active(table, obj_id, not active)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        st.caption(f"{row.get(name_col)} - ID {obj_id}" + (" - default" if is_default else ""))


def _delete_body(pipeline: AstroPipeline, table: str, obj_id: int) -> None:
    """Content of the Delete window."""
    spec = _SPECS[table]
    one, name_col = spec["one"], spec["name"]
    db = pipeline.db
    row = _row(_load(pipeline, table), obj_id)
    name = row.get(name_col)
    kp = f"ref_{table}_{obj_id}"
    is_default = int(row.get("IS_DEFAULT") or 0) == 1
    active = row.get("ACTIVE") == "YES"
    used = db.reference_object_usage(table, obj_id)
    if is_default:
        used = [*used, f"it is the default {one} (pre-selected at import)"]
    extra = " together with its gain-curve points" if table == "EQUIPMENTS" else ""
    if not used:
        st.write(f"Nothing uses this {one}. It is removed from the database{extra}. This cannot be undone.")
        c1, c2 = st.columns(2)
        if c1.button("Cancel", key=f"{kp}_dcancel", width="stretch"):
            st.rerun()
        if c2.button("Delete", key=f"{kp}_dok", type="primary", width="stretch"):
            try:
                db.delete_reference_object(table, obj_id)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        return
    st.warning(f"This {one} is used by:\n\n" + "\n".join(f"- {u}" for u in used))
    st.write("Recommended: **Hide** it - it disappears from the import pickers and stays with that data.")
    st.error(
        "Delete anyway: calibration masters lose the link and are no longer used; processed fields keep their files, but a "
        f"new run of them has to choose the {one} again."
        + (f" The default passes to the active {one} with the lowest ID." if is_default else "")
    )
    ok = st.checkbox(f"I understand - delete {name} anyway", key=f"{kp}_force_ok")
    c1, c2, c3 = st.columns(3)
    if c1.button("Cancel", key=f"{kp}_dcancel", width="stretch"):
        st.rerun()
    if c2.button(f"Hide {one}", key=f"{kp}_dhide", width="stretch", disabled=is_default or not active):
        db.set_reference_active(table, obj_id, False)
        st.rerun()
    if c3.button("Delete anyway", key=f"{kp}_force", disabled=not ok, width="stretch"):
        try:
            db.delete_reference_object(table, obj_id, force=True)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


def _open_edit(pipeline: AstroPipeline, table: str, obj_id: int | None, name: str = "") -> None:
    one = _SPECS[table]["one"]
    title = f"Edit {one} - {name}" if obj_id is not None else f"Add {one}"
    st.dialog(title, width="large")(_edit_body)(pipeline, table, obj_id)


def _open_delete(pipeline: AstroPipeline, table: str, obj_id: int, name: str) -> None:
    st.dialog(f"Delete {_SPECS[table]['one']} {name}?")(_delete_body)(pipeline, table, obj_id)


def _render_objects(pipeline: AstroPipeline, table: str) -> None:
    """REF-OBJECTS-2026-10: table of cameras / telescopes / sites with Edit and Delete on every row."""
    spec = _SPECS[table]
    one, name_col = spec["one"], spec["name"]
    df = _load(pipeline, table)
    if table == "EQUIPMENTS" and st.session_state.get("ref_added_camera") is not None:
        _new = int(st.session_state.pop("ref_added_camera"))
        _r = _row(df, _new)
        _open_edit(pipeline, table, _new, str(_r.get(name_col) or ""))  # enter the gain setting you use
    t1, t2 = st.columns([4, 1])
    show_hidden = t1.toggle(f"Show hidden {one}s", value=False, key=f"ref_hidden_{table}")
    if t2.button(f"+ Add {one}", key=f"ref_add_{table}", type="primary", width="stretch"):
        _open_edit(pipeline, table, None)
    view = df if show_hidden or df.empty else df[df["ACTIVE"] == "YES"]
    if view.empty:
        st.info(f"No {one} yet - add one with + Add {one}.")
        return
    cols = [c for c in spec["list"] if c[0] in view.columns]
    widths = [w for _, _, w in cols] + [0.9, 2.0]
    with st.container(border=True):
        hdr = st.columns(widths, vertical_alignment="center")
        for c, (_, label, _) in zip(hdr, cols):
            c.markdown(f"**{label}**")
        hdr[-1].markdown("**Actions**")
        for r in view.to_dict("records"):
            st.markdown('<hr style="margin:0;border:none;border-top:1px solid rgba(128,128,128,0.25)">',
                        unsafe_allow_html=True)
            rc = st.columns(widths, vertical_alignment="center")
            hidden = r.get("ACTIVE") != "YES"
            for c, (col, _, _) in zip(rc, cols):
                txt = _fmt_cell(col, r.get(col))
                c.markdown(f":gray[{txt}]" if hidden or txt == "-" else txt)
            tag = "default" if int(r.get("IS_DEFAULT") or 0) == 1 else ("hidden" if hidden else "")
            if tag:
                rc[-2].markdown(f":red-badge[{tag}]" if tag == "default" else f":orange-badge[{tag}]")
            a1, a2 = rc[-1].columns(2)
            rid = int(r["ID"])
            if a1.button("Edit", key=f"ref_edit_{table}_{rid}", width="stretch"):
                _open_edit(pipeline, table, rid, str(r.get(name_col)))
            if a2.button("Delete", key=f"ref_del_{table}_{rid}", width="stretch"):
                _open_delete(pipeline, table, rid, str(r.get(name_col)))


def _render_add_from_catalog(pipeline: AstroPipeline) -> None:
    from camera_profile import catalog_mode, catalog_model, catalog_models  # noqa: PLC0415

    models = catalog_models()
    if not models:
        st.info("The camera catalog is empty - use the empty form.")
        return
    model_labels = {_model_label(m): m["key"] for m in models}
    mk = model_labels[st.selectbox("Camera model", list(model_labels), key="vyvar_add_model")]
    m = catalog_model(mk) or {}
    mode_labels = {str(md.get("name") or md.get("key")): str(md.get("key"))
                   for md in m.get("modes", []) if isinstance(md, dict)}
    mo = mode_labels[st.selectbox("Readout mode", list(mode_labels), key=f"vyvar_add_mode_{mk}")]
    name = st.text_input("Name in VYVAR", value=str(m.get("model") or ""), key=f"vyvar_add_name_{mk}")
    md = catalog_mode(m, mo) or {}
    st.caption(
        f"Fills: sensor {m.get('sensor')}, {md.get('width')}x{md.get('height')} px of {md.get('pixel_um')} um, "
        f"{md.get('adc_bits')}-bit ADC, binning {md.get('binning') or 'not stated (average assumed)'}, "
        f"header GAIN = {m.get('header_gain')}, {len(md.get('curve') or [])} catalog curve point(s). "
        "Afterwards enter the gain setting you use and its gain / read noise, or measure them "
        "(Calibration Library > Measure gain)."
    )
    if st.button("Add camera", key="vyvar_add_btn", type="primary"):
        try:
            new_id = pipeline.db.add_equipment_from_catalog(mk, mo, name)
            st.session_state["ref_added_camera"] = int(new_id)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


def render_database_explorer(pipeline: AstroPipeline) -> None:
    st.subheader("Database Explorer")
    st.caption(
        "Your cameras, telescopes and observing sites. Add, edit, hide or delete them; the default of each "
        "is pre-selected when you import a night. A hidden object stays with the data that use it."
    )
    t_cam, t_tel, t_loc = st.tabs(["Cameras", "Telescopes", "Sites"])
    with t_cam:
        _render_cameras(pipeline)
    with t_tel:
        _render_objects(pipeline, "TELESCOPE")
    with t_loc:
        _render_objects(pipeline, "LOCATION")
