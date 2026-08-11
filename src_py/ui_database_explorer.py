"""Database Explorer tab: table browser + staging maintenance (OBS_FILES / OBS_DRAFT only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from database import VyvarDatabase
from draft_provenance import (
    collect_manifest_draft_rows,
    collect_manifest_obs_file_rows,
)
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


def _render_obs_draft_manifest_editor(pipeline: AstroPipeline, archive_root: Path) -> None:
    """OBS_DRAFT editor: display from manifest; save via SQL + manifest refresh."""
    draft_rows = collect_manifest_draft_rows(archive_root)
    draft_df = pd.DataFrame.from_records(draft_rows) if draft_rows else pd.DataFrame()
    if draft_df.empty:
        st.info("No draft folders found under Archive/Drafts.")
        return
    if "ID" in draft_df.columns:
        draft_df = draft_df.sort_values("ID", ascending=False).reset_index(drop=True)
    st.caption("Source: ``draft_manifest.json``. Save writes OBS_DRAFT SQL then refreshes manifest.")
    editable_cols = [
        "ID_EQUIPMENTS",
        "ID_TELESCOPE",
        "ID_LOCATION",
        "ID_SCANNING",
        "LIGHTS_PATH",
        "CALIB_PATH",
        "ARCHIVE_PATH",
        "MASTERSTAR_PATH",
        "MASTERSTAR_FITS_PATH",
        "STATUS",
        "CENTEROFFIELDRA",
        "CENTEROFFIELDDE",
        "OBSERVATIONSTARTJD",
        "IS_CALIBRATED",
    ]
    editable = [c for c in editable_cols if c in draft_df.columns]
    disabled = [c for c in draft_df.columns if c == "ID" or c not in editable]
    edited = st.data_editor(
        draft_df,
        width="stretch",
        num_rows="dynamic",
        disabled=disabled,
        key="vyvar_universal_ed_OBS_DRAFT",
        hide_index=True,
    )
    if st.button("Save changes to database (OBS_DRAFT manifest)", key="vyvar_universal_save_OBS_DRAFT"):
        try:
            stats = pipeline.db.apply_main_table_editor_save(
                "OBS_DRAFT",
                "ID",
                draft_df,
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
                parts.append(f"soft-deactivated (ACTIVE='NO'): {sd}")
            st.success("Done: " + ", ".join(parts) + ".")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


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
    df = _read_df(conn, f"SELECT * FROM {sql_name} {order_sql};")
    if "ACTIVE" in df.columns and not df.empty:
        df["ACTIVE"] = df["ACTIVE"].map(VyvarDatabase.normalize_active_text)
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
            "**TELESCOPE.ACTIVE:** **YES** = active; **NO** = soft-delete (inactive). "
            "Deleting a row in the editor **does not remove** the record - sets **ACTIVE = 'NO'**."
        )
    elif sql_name == "EQUIPMENTS":
        st.caption(
            "**EQUIPMENTS.ACTIVE:** **YES** = active; **NO** = soft-delete (only active rows in Draft picker). "
            "Deleting a row in the editor **does not DELETE** in SQL - sets **ACTIVE = 'NO'**. "
            "Physical ``DELETE`` is not performed from this editor (``FINAL_DATA`` / hash integrity)."
        )
    else:
        st.caption(
            "After **Save**, changes are written to SQL. For **LOCATION**, a row is **deleted** "
            "only if nothing in OBS_DRAFT references it (otherwise an error)."
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
    if "ACTIVE" in df.columns and sql_name in ("EQUIPMENTS", "TELESCOPE", "LOCATION"):
        column_config["ACTIVE"] = st.column_config.SelectboxColumn(
            "ACTIVE",
            options=["YES", "NO"],
            help="YES = active, NO = soft-delete (inactive)",
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
                parts.append(f"soft-deactivated (ACTIVE='NO'): {sd}")
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
        options=["TELESCOPES", "EQUIPMENTS", "LOCATION", "OBS_DRAFT", "OBS_FILES"],
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

    elif table == "OBS_DRAFT":
        archive_root = Path(pipeline.config.archive_root)
        _render_obs_draft_manifest_editor(pipeline, archive_root)
        st.info("OBS_DRAFT rows represent ingestion before astrometry finalization.")

    elif table == "OBS_FILES":
        st.caption("Per-file index from ``draft_manifest.json`` ``files[]`` (ingestion evidence).")
        archive_root = Path(pipeline.config.archive_root)
        obs_rows = collect_manifest_obs_file_rows(archive_root)
        obs_ids = sorted(
            {
                str(r["OBSERVATION_ID"])
                for r in obs_rows
                if r.get("OBSERVATION_ID") not in (None, "")
            }
        )[:200]
        draft_ids = [str(r["ID"]) for r in collect_manifest_draft_rows(archive_root)[:200]]
        selected_obs = st.selectbox(
            "Filter by key",
            options=["(all)"] + [f"OBS:{x}" for x in obs_ids] + [f"DRAFT:{x}" for x in draft_ids],
            index=0,
        )
        if selected_obs == "(all)":
            file_rows = collect_manifest_obs_file_rows(archive_root)[:2000]
        elif selected_obs.startswith("DRAFT:"):
            did = int(selected_obs.split(":", 1)[1])
            file_rows = collect_manifest_obs_file_rows(archive_root, draft_id=did)
        else:
            oid = selected_obs.split(":", 1)[1] if selected_obs.startswith("OBS:") else selected_obs
            file_rows = collect_manifest_obs_file_rows(archive_root, observation_id=oid)
        files_df = pd.DataFrame.from_records(file_rows) if file_rows else pd.DataFrame()
        st.dataframe(files_df, width="stretch")
        st.info("OBS_FILES edit is disabled (generated automatically during import).")

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
