"""Database Explorer tab: table browser + staging maintenance (OBS_FILES / OBS_DRAFT only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from database import VyvarDatabase
from importer import quicklook_preview_png_bytes
from pipeline import AstroPipeline


def _row_active_for_style(row: pd.Series, col: str = "ACTIVE") -> bool:
    if col not in row.index:
        return True
    v = row.get(col)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return True
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            return int(v) != 0
        except (TypeError, ValueError):
            return True
    s = str(v).strip().upper()
    if s in ("NO", "N", "FALSE", "0", "0.0"):
        return False
    return True


def _render_universal_main_table(
    pipeline: AstroPipeline,
    *,
    sql_name: str,
    ui_label: str,
    editable_cols: list[str],
    order_sql: str = "ORDER BY ID",
    extra_caption: str | None = None,
) -> None:
    """``st.data_editor`` with dynamic rows + Save -> SQL (non-OBS reference tables)."""
    conn = pipeline.db.conn
    df = pd.read_sql_query(f"SELECT * FROM {sql_name} {order_sql};", conn)
    if sql_name == "TELESCOPE" and "ACTIVE" in df.columns and not df.empty:
        df["ACTIVE"] = df["ACTIVE"].map(VyvarDatabase.normalize_active_db_value).astype(int)
    if "IS_DEFAULT" in df.columns and not df.empty:
        df["IS_DEFAULT"] = (
            df["IS_DEFAULT"].map(VyvarDatabase.normalize_active_db_value).astype(int).astype(bool)
        )
    editable = [c for c in editable_cols if c in df.columns]
    if sql_name in ("EQUIPMENTS", "TELESCOPE", "LOCATION") and "IS_DEFAULT" in df.columns:
        st.caption(
            "**IS_DEFAULT:** exactly one row is the default (pre-selected in Scan Source). "
            "Check a different row and **Save** to move the default - the previous one is cleared automatically."
        )
    if sql_name == "TELESCOPE":
        st.caption(
            "**TELESCOPE.ACTIVE:** only **0** or **1** - **1** = active, **0** = soft-delete (inactive). "
            "Deleting a row in the editor **does not remove** the record - sets **ACTIVE = 0**."
        )
    elif sql_name == "EQUIPMENTS":
        st.caption(
            "**EQUIPMENTS.ACTIVE:** **1** / **YES** = active; **0** / **NO** = soft-delete (only active rows in Draft picker). "
            "Deleting a row in the editor **does not DELETE** in SQL - sets **ACTIVE = 0**. "
            "Physical ``DELETE`` is not performed from this editor (``FINAL_DATA`` / hash integrity)."
        )
    else:
        st.caption(
            "After **Save**, changes are written to SQL. For **LOCATION** / **SCANNING**, a row is **deleted** "
            "only if nothing in OBS_DRAFT / OBSERVATION references it (otherwise an error)."
        )
    if extra_caption:
        st.caption(extra_caption)
    if "ACTIVE" in df.columns and not df.empty:

        def _grey_inactive(r: pd.Series) -> list[str]:
            ok = _row_active_for_style(r, "ACTIVE")
            return ["" if ok else "color: #6c757d; text-decoration: line-through" for _ in r.index]

        with st.expander("Preview - inactive rows are gray", expanded=False):
            st.dataframe(df.style.apply(_grey_inactive, axis=1), width="stretch")

    disabled = [c for c in df.columns if c == "ID" or c not in editable]
    column_config: dict[str, Any] = {}
    if sql_name == "TELESCOPE" and "ACTIVE" in df.columns:
        column_config["ACTIVE"] = st.column_config.SelectboxColumn(
            "ACTIVE",
            options=[0, 1],
            help="1 = active, 0 = soft-delete (inactive)",
            required=True,
        )
    if "IS_DEFAULT" in df.columns:
        column_config["IS_DEFAULT"] = st.column_config.CheckboxColumn(
            "IS_DEFAULT",
            help="Exactly one default per table; pre-selected in Scan Source.",
        )
    edited = st.data_editor(
        df,
        width="stretch",
        num_rows="dynamic",
        disabled=disabled,
        column_config=column_config if column_config else None,
        key=f"vyvar_universal_ed_{sql_name}",
        hide_index=True,
    )
    if st.button(f"Save changes to database ({ui_label})", key=f"vyvar_universal_save_{sql_name}"):
        try:
            stats = pipeline.db.apply_main_table_editor_save(
                sql_name,
                "ID",
                df,
                edited,
                editable_cols=editable,
            )
            sd = int(stats.get("soft_deactivated", 0))
            parts = [
                f"inserted {stats['inserted']}",
                f"updated {stats['updated']}",
                f"deleted {stats['deleted']}",
            ]
            if sd:
                parts.append(f"soft-deactivated (ACTIVE=0): {sd}")
            st.success("Done: " + ", ".join(parts) + ".")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


def render_database_explorer(pipeline: AstroPipeline) -> None:
    st.subheader("Database Explorer")
    st.caption("Browse and validate VYVAR SQLite metadata (with basic consistency tools).")

    conn = pipeline.db.conn

    table = st.selectbox(
        "Table",
        options=["TELESCOPES", "EQUIPMENTS", "LOCATION", "SCANNING", "OBS_DRAFT", "OBSERVATION", "OBS_FILES"],
        index=0,
    )

    if table == "TELESCOPES":
        _render_universal_main_table(
            pipeline,
            sql_name="TELESCOPE",
            ui_label="TELESCOPE",
            editable_cols=["TELESCOPENAME", "ALIAS", "DIAMETER", "FOCAL", "ACTIVE", "IS_DEFAULT"],
        )

    elif table == "EQUIPMENTS":
        _render_universal_main_table(
            pipeline,
            sql_name="EQUIPMENTS",
            ui_label="EQUIPMENTS",
            editable_cols=[
                "CAMERANAME",
                "ALIAS",
                "SENSORTYPE",
                "SENSORSIZE",
                "PIXELSIZE",
                "FOCAL",
                "SATURATE_ADU",
                "GAIN_ADU",
                "READNOISE_E",
                "BAYERMASK",
                "ACTIVE",
                "IS_DEFAULT",
            ],
            extra_caption=(
                "SATURATE_ADU: ADU ceiling for catalog saturation. "
                "BAYERMASK: RGGB/BGGR/GBRG/GRBG for OSC; empty or mono for monochrome. "
                "E.g. 16383 for 14-bit ADC, or 65535 for full 16-bit range."
            ),
        )

    elif table == "LOCATION":
        _render_universal_main_table(
            pipeline,
            sql_name="LOCATION",
            ui_label="LOCATION",
            editable_cols=["PLACENAME", "LATITUDE", "LONGITUDE", "ALTITUDE", "ACTIVE", "IS_DEFAULT"],
        )

    elif table == "SCANNING":
        st.caption(
            "Reference table for exposure / filter / binning. Editor changes apply only to SCANNING columns."
        )
        with st.expander("Preview: Effective_Pixel_Size (reference EQUIPMENTS ID=1)", expanded=False):
            scan_preview = pd.read_sql_query(
                """
                SELECT
                    s.*,
                    e.PIXELSIZE AS Base_Pixel_Size_um,
                    CASE
                        WHEN s.BINNING = 22 THEN e.PIXELSIZE * 2.0
                        ELSE e.PIXELSIZE
                    END AS Effective_Pixel_Size_um
                FROM SCANNING s
                LEFT JOIN EQUIPMENTS e ON e.ID = 1
                ORDER BY s.ID;
                """,
                conn,
            )
            if not scan_preview.empty and "BINNING" in scan_preview.columns:

                def _highlight_binning_22(row: pd.Series) -> list[str]:
                    try:
                        is_22 = int(row["BINNING"]) == 22
                    except Exception:  # noqa: BLE001
                        is_22 = False
                    return ["background-color: #fff3cd" if is_22 else "" for _ in row.index]

                st.dataframe(
                    scan_preview.style.apply(_highlight_binning_22, axis=1),
                    width="stretch",
                )
            else:
                st.dataframe(scan_preview, width="stretch")

        _scan_editable = ["EXPTIME", "FILTERS", "BINNING", "SENSORTEMP", "GAIN"]
        _render_universal_main_table(
            pipeline,
            sql_name="SCANNING",
            ui_label="SCANNING",
            editable_cols=_scan_editable,
        )

    elif table == "OBS_DRAFT":
        draft_df = pd.read_sql_query("SELECT * FROM OBS_DRAFT ORDER BY ID DESC;", conn)
        st.dataframe(draft_df, width="stretch")
        st.info("OBS_DRAFT rows represent ingestion before astrometry finalization.")

    elif table == "OBS_FILES":
        st.caption("Per-file index for each OBSERVATION (ingestion evidence).")
        obs_ids = pd.read_sql_query(
            "SELECT ID FROM OBSERVATION ORDER BY ID DESC LIMIT 200;",
            conn,
        )["ID"].astype(str).tolist()
        draft_ids = pd.read_sql_query(
            "SELECT ID FROM OBS_DRAFT ORDER BY ID DESC LIMIT 200;",
            conn,
        )["ID"].astype(str).tolist()
        selected_obs = st.selectbox(
            "Filter by key",
            options=["(all)"] + [f"OBS:{x}" for x in obs_ids] + [f"DRAFT:{x}" for x in draft_ids],
            index=0,
        )
        if selected_obs == "(all)":
            files_df = pd.read_sql_query(
                "SELECT * FROM OBS_FILES ORDER BY OBSERVATION_ID DESC, ID DESC LIMIT 2000;",
                conn,
            )
        elif selected_obs.startswith("DRAFT:"):
            did = selected_obs.split(":", 1)[1]
            files_df = pd.read_sql_query(
                "SELECT * FROM OBS_FILES WHERE DRAFT_ID = ? ORDER BY ID DESC;",
                conn,
                params=(did,),
            )
        else:
            oid = selected_obs.split(":", 1)[1] if selected_obs.startswith("OBS:") else selected_obs
            files_df = pd.read_sql_query(
                "SELECT * FROM OBS_FILES WHERE OBSERVATION_ID = ? ORDER BY ID DESC;",
                conn,
                params=(oid,),
            )
        st.dataframe(files_df, width="stretch")
        st.info("OBS_FILES edit is disabled (generated automatically during import).")

    else:  # OBSERVATION
        observation_df = pd.read_sql_query("SELECT * FROM OBSERVATION ORDER BY ID;", conn)

        if "IS_CALIBRATED" in observation_df.columns:
            draft_mask = observation_df["IS_CALIBRATED"].fillna(1).astype(int) == 0
            if draft_mask.any():
                st.error(
                    "Draft sessions detected: some OBSERVATION rows are marked as non-calibrated. "
                    "These sessions should be treated as preliminary (Quick Look)."
                )

                def _highlight_draft(row: pd.Series) -> list[str]:
                    is_draft = False
                    try:
                        is_draft = int(row.get("IS_CALIBRATED", 1) or 1) == 0
                    except Exception:  # noqa: BLE001
                        is_draft = False
                    return ["background-color: #f8d7da" if is_draft else "" for _ in row.index]

                st.dataframe(
                    observation_df.style.apply(_highlight_draft, axis=1),
                    width="stretch",
                )
            else:
                st.dataframe(observation_df, width="stretch")
        else:
            st.dataframe(observation_df, width="stretch")

        # Quick Look preview (ZScale) for draft sessions
        if "IS_CALIBRATED" in observation_df.columns and "LIGHTS_PATH" in observation_df.columns:
            draft_df = observation_df[
                observation_df["IS_CALIBRATED"].fillna(1).astype(int) == 0
            ].copy()
            if not draft_df.empty:
                st.markdown("---")
                st.subheader("Quick Look Preview (Draft)")
                choices = draft_df["ID"].astype(str).tolist()
                selected = st.selectbox("Draft OBSERVATION", options=choices)
                row = draft_df[draft_df["ID"].astype(str) == str(selected)].iloc[0]
                lights_path = str(row.get("LIGHTS_PATH") or "")
                lights_dir = Path(lights_path)
                fits_files: list[Path] = []
                if lights_dir.exists() and lights_dir.is_dir():
                    for ext in ("*.fits", "*.fit", "*.fts", "*.FITS", "*.FIT", "*.FTS"):
                        fits_files.extend(lights_dir.glob(ext))
                fits_files = sorted(fits_files)
                if not fits_files:
                    st.warning("No FITS found for preview in LIGHTS_PATH.")
                else:
                    try:
                        png_bytes = quicklook_preview_png_bytes(fits_files[0])
                        st.image(png_bytes, caption=f"ZScale preview: {fits_files[0].name}", width="stretch")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Preview failed: {exc}")

        st.info("OBSERVATION edit is intentionally disabled (ID is derived / file-first key).")

    st.divider()
    st.subheader("Database Maintenance (Temporary Tables Only)")
    st.caption(
        "This section works **only** with **OBS_FILES** and **OBS_DRAFT**. "
        "It does not run SQL against **EQUIPMENTS**, **TELESCOPE**, or **OBS_QC_PROCESSING_*** (final hashes)."
    )
    _n_obs = pipeline.db.count_obs_files()
    st.metric("Row count in OBS_FILES", int(_n_obs))

    st.warning(
        "Really delete temporary data for processed observations? "
        "Rows in `OBS_FILES` belonging to drafts with status **PROCESSED** will be removed."
    )
    if st.button("Cleanup Processed Data", key="vyvar_dbx_maint_cleanup_processed"):
        try:
            _del = pipeline.db.maintenance_delete_obs_files_for_processed_drafts()
            st.success(f"Deleted rows in OBS_FILES: {_del}.")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
        st.rerun()

    st.markdown(
        '<p style="color:#c0392b;font-weight:700;margin-top:1.25rem;">Nuke Staging Area (Danger Zone)</p>',
        unsafe_allow_html=True,
    )
    st.error(
        "Full staging reset: **DELETE FROM OBS_FILES** and **DELETE FROM OBS_DRAFT**. "
        "Hashtag tables **OBS_QC_PROCESSING_*** are left intact (may still reference missing draft IDs)."
    )
    if "vyvar_maint_nuke_gen" not in st.session_state:
        st.session_state["vyvar_maint_nuke_gen"] = 0
    _nuke_gen = int(st.session_state["vyvar_maint_nuke_gen"])
    _nuke_ok = st.checkbox(
        "I understand the risks and want to delete all rows in OBS_FILES and OBS_DRAFT.",
        key=f"vyvar_dbx_maint_nuke_confirm_{_nuke_gen}",
    )
    _nuke_clicked = st.button(
        ":red[Nuke Staging Area (Danger Zone)]",
        key=f"vyvar_dbx_maint_nuke_go_{_nuke_gen}",
        disabled=not _nuke_ok,
        type="primary",
        help="Deletes all rows in OBS_FILES and OBS_DRAFT. EQUIPMENTS / TELESCOPE / OBS_QC_PROCESSING_* are unchanged.",
    )
    if _nuke_clicked:
        try:
            nf, nd = pipeline.db.maintenance_nuke_obs_files_and_drafts_preserve_qc_snapshots()
            st.success(f"Deleted rows: OBS_FILES = {nf}, OBS_DRAFT = {nd}.")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
        st.session_state["vyvar_maint_nuke_gen"] = _nuke_gen + 1
        st.rerun()
