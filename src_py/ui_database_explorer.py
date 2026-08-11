"""Database Explorer tab: TELESCOPE / EQUIPMENTS / LOCATION reference-table editors."""

from __future__ import annotations

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


def _render_universal_main_table(
    pipeline: AstroPipeline,
    *,
    sql_name: str,
    ui_label: str,
    editable_cols: list[str],
    order_sql: str = "ORDER BY ID",
    extra_caption: str | None = None,
) -> None:
    """``st.data_editor`` with dynamic rows + Save -> SQL (reference tables only)."""
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
            "only if nothing in draft manifests references it (otherwise an error)."
        )
    if extra_caption:
        st.caption(extra_caption)

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

    table = st.selectbox(
        "Table",
        options=["TELESCOPES", "EQUIPMENTS", "LOCATION"],
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
