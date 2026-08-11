"""UI: final approval step - persist OBSERVATION and archive key artifacts under ``finalized/``."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from astropy.io import fits

from infolog import log_event
from draft_provenance import draft_scan_summary_from_manifest, load_draft_manifest

# Columns read from per-frame *_catalog.csv for FIELD_REGISTRY sidecar RMS (groupby only).
_REGISTRY_COLS = ["catalog_id", "aperture_mag"]


def _copy_finalization_files(
    archive_path: Path,
    draft_id: int,
    observation_id: str,
) -> list[str]:
    """Copy platesolve artifacts and per-frame ``*_catalog.csv`` into ``archive_path/finalized/``."""
    _ = draft_id, observation_id
    copied: list[str] = []
    try:
        root = Path(archive_path).resolve()
        fin = root / "finalized"
        fin.mkdir(parents=True, exist_ok=True)
        ps = root / "platesolve"
        pairs = [
            (ps / "MASTERSTAR.fits", fin / "MASTERSTAR.fits"),
            (ps / "masterstars_full_match.csv", fin / "masterstars_full_match.csv"),
            (ps / "comparison_stars.csv", fin / "comparison_stars.csv"),
            (ps / "variable_targets.csv", fin / "variable_targets.csv"),
            (ps / "masterstar_epsf.fits", fin / "masterstar_epsf.fits"),
            (ps / "masterstar_epsf_meta.json", fin / "masterstar_epsf_meta.json"),
        ]
        for src, dst in pairs:
            try:
                if src.is_file():
                    shutil.copy2(src, dst)
                    copied.append(dst.name)
                else:
                    log_event(f"Finalizacia: preskoceny chybajuci subor {src}")
            except Exception as exc:  # noqa: BLE001
                # EXC-0512: T3 -- UI diagnostic/plot only (else: / log_event(f'Finalizacia: preskoceny chybajuci subor {s... (EXCEPT-BULK 2026-07-08)
                log_event(f"Finalizacia: kopirovanie zlyhalo {src} -> {dst}: {exc!s}")

        pcsv = root / "processed"
        if pcsv.is_dir():
            out_flat = fin / "per_frame_csv"
            try:
                out_flat.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                # EXC-0513: T3 -- UI diagnostic/plot only (try: / out_flat.mkdir(parents=True, exist_ok=True) / except Ex... (EXCEPT-BULK 2026-07-08)
                log_event(f"Finalizacia: nemozem vytvorit {out_flat}: {exc!s}")
            else:
                for src in pcsv.rglob("*_catalog.csv"):
                    try:
                        if src.is_file():
                            dst = out_flat / src.name
                            shutil.copy2(src, dst)
                            copied.append(f"per_frame_csv/{dst.name}")
                    except Exception as exc:  # noqa: BLE001
                        # EXC-0514: T3 -- UI diagnostic/plot only (shutil.copy2(src, dst) / copied.append(f'per_frame_csv/{dst.na... (EXCEPT-BULK 2026-07-08)
                        log_event(f"Finalizacia: kopirovanie zlyhalo {src}: {exc!s}")
    except Exception as exc:  # noqa: BLE001
        # EXC-0515: T3 -- UI diagnostic/plot only (except Exception as exc:  # noqa: BLE001 / log_event(f'Finaliz... (EXCEPT-BULK 2026-07-08)
        log_event(f"Finalizacia: _copy_finalization_files: {exc!s}")
    return copied


def _draft_scan_row(db: Any, draft_id: int, archive_path: Path | None) -> dict[str, Any] | None:
    if archive_path is not None:
        manifest = load_draft_manifest(archive_path)
        if manifest:
            return draft_scan_summary_from_manifest(manifest)
    row = db.fetch_obs_draft_by_id(int(draft_id))
    if row is None:
        return None
    ap_raw = row.get("ARCHIVE_PATH")
    if ap_raw:
        manifest = load_draft_manifest(str(ap_raw))
        if manifest:
            return draft_scan_summary_from_manifest(manifest)
    return None


def _draft_location_name(db: Any, draft_id: int) -> str:
    try:
        loc_id = db.get_draft_location_id(int(draft_id))
        if loc_id is None:
            return "-"
        row = db.conn.execute(
            "SELECT PLACENAME FROM LOCATION WHERE ID = ?;",
            (int(loc_id),),
        ).fetchone()
        if row is None:
            return "-"
        v = row["PLACENAME"] if hasattr(row, "keys") else row[0]
        s = str(v).strip() if v is not None else ""
        return s or "-"
    except Exception:  # noqa: BLE001
        return "-"


def _n_light_frames(db: Any, draft_id: int) -> int:
    try:
        rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
        return len(rows)
    except Exception:  # noqa: BLE001
        return 0


def _csv_nonempty(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        df = pd.read_csv(path, nrows=5000)
        return len(df) > 0
    except Exception:  # noqa: BLE001
        # EXC-0519: T3 -- UI diagnostic/plot only (df = pd.read_csv(path, nrows=5000) / return len(df) > 0 / exce... (EXCEPT-BULK 2026-07-08)
        return False


# NOTE: render_finalization is not wired into app.py (inactive).
# render_known_field_banner() below is active (app.py).
# Last audit: 2026-05-18
def render_finalization(
    pipeline: Any,
    draft_id: int | None,
) -> None:
    st.subheader("Observation finalization")
    st.caption(
        "Review all tabs before approval. After approval, the observation is saved permanently."
    )

    if draft_id is None:
        st.warning("Select or process a draft (no observation selected).")
        return

    db = pipeline.db
    row = db.fetch_obs_draft_by_id(int(draft_id))
    if row is None:
        st.error(f"Draft {draft_id} was not found in the database.")
        return

    arch_raw = row.get("ARCHIVE_PATH")
    try:
        archive_path = Path(str(arch_raw)).expanduser().resolve() if arch_raw else None
    except OSError:
        archive_path = None

    st.markdown("#### Checklist")
    checks: list[tuple[str, bool]] = []

    ms_path = (archive_path / "platesolve" / "MASTERSTAR.fits") if archive_path else None
    checks.append(("MASTERSTAR exists", bool(ms_path and ms_path.is_file())))

    plate_ok = False
    if ms_path and ms_path.is_file():
        try:
            with fits.open(ms_path, memmap=False) as hdul:
                h0 = hdul[0].header
                plate_ok = ("VY_SIPRF" in h0) or ("VY_PSOLV" in h0)
        except Exception:  # noqa: BLE001
            plate_ok = False
    checks.append(("Plate solve OK (VY_SIPRF or VY_PSOLV in MASTERSTAR)", plate_ok))

    n_cat = 0
    if archive_path and (archive_path / "processed").is_dir():
        n_cat = sum(1 for _ in (archive_path / "processed").rglob("*_catalog.csv"))
    checks.append(("Per-frame CSV exists (at least 1 x *_catalog.csv in processed/)", n_cat >= 1))

    comp_path = (archive_path / "platesolve" / "comparison_stars.csv") if archive_path else None
    checks.append(
        ("Comp stars defined (comparison_stars.csv, >0 rows)", _csv_nonempty(comp_path) if comp_path else False)
    )

    var_path = (archive_path / "platesolve" / "variable_targets.csv") if archive_path else None
    checks.append(
        (
            "Variable targets defined (variable_targets.csv, >0 rows)",
            _csv_nonempty(var_path) if var_path else False,
        )
    )

    st_fin = str(row.get("STATUS") or "").strip().upper()
    checks.append(("Draft is not already finalized", st_fin != "FINALIZED"))

    for label, ok in checks:
        st.markdown(f"{'[OK]' if ok else '!'} {label}")

    if not all(c[1] for c in checks):
        st.warning("Some checks failed - you may still continue (your call).")

    st.markdown("#### Observation summary")
    scan = _draft_scan_row(db, int(draft_id), archive_path)
    loc_name = _draft_location_name(db, int(draft_id))
    tel_eq = db.fetch_obs_draft_telescope_equipment(int(draft_id)) or {}
    n_frames = _n_light_frames(db, int(draft_id))

    ra_v = row.get("CENTEROFFIELDRA")
    de_v = row.get("CENTEROFFIELDDE")
    ra_s = f"{float(ra_v):.6f} deg" if ra_v is not None else "-"
    de_s = f"{float(de_v):.6f} deg" if de_v is not None else "-"
    obj = str(row.get("OBJECT") or "").strip() or "-"
    flt = str((scan or {}).get("filters") or "").strip() or "-"
    expt = (scan or {}).get("exptime")
    expt_s = f"{float(expt):.2f} s" if expt is not None else "-"
    binv = (scan or {}).get("binning")
    bin_s = str(int(binv)) if binv is not None else "-"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Object:** {obj}")
        st.markdown(f"**RA / Dec:** {ra_s} / {de_s}")
        st.markdown(f"**Filter:** {flt}")
        st.markdown(f"**Exposure:** {expt_s}")
        st.markdown(f"**Binning:** {bin_s}")
        st.markdown(f"**Light frame count (manifest files[]):** {n_frames}")
    with c2:
        st.markdown(f"**DATE_OBS start:** {row.get('DATE_OBS_START') or '-'}")
        st.markdown(f"**DATE_OBS end:** {row.get('DATE_OBS_END') or '-'}")
        st.markdown(f"**Equipment:** {tel_eq.get('equipment_name') or '-'}")
        st.markdown(f"**Telescope:** {tel_eq.get('telescope_name') or '-'}")
        st.markdown(f"**Location:** {loc_name}")
        st.markdown(f"**Archive path:** `{archive_path or '-'}`")

    st.markdown("#### Approval")
    approved_by = st.text_input(
        "Observer name (written to OBSERVATION)",
        value=st.session_state.get("vyvar_observer_name", ""),
        key="vyvar_finalization_approved_by",
    )
    notes = st.text_area(
        "Observation notes (optional)",
        key="vyvar_finalization_notes",
        height=80,
    )

    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(
            "After approval:\n"
            "1. The observation is written to the OBSERVATION table\n"
            "2. Important files are copied to finalized/\n"
            "3. The draft remains in the DB (manual cleanup in Database Explorer)"
        )
    with col2:
        confirm = st.checkbox(
            "I understand, finalize now",
            key="vyvar_finalization_confirm",
        )
        finalize_btn = st.button(
            "[OK] Approve and finalize",
            type="primary",
            disabled=not confirm,
            key="vyvar_finalization_go",
        )

    if finalize_btn and confirm:
        with st.spinner("Finalizing observation..."):
            try:
                obs_id = pipeline.db.finalize_draft_to_observation(
                    int(draft_id),
                    approved_by=approved_by.strip() or None,
                    notes=notes.strip() or None,
                )
                log_event(f"Finalizacia: draft {draft_id} -> OBSERVATION {obs_id}")

                if archive_path is None:
                    st.warning("ARCHIVE_PATH is not set - skipping file copy.")
                    copied: list[str] = []
                else:
                    copied = _copy_finalization_files(archive_path, int(draft_id), obs_id)
                log_event(f"Finalizacia: skopirovanych {len(copied)} suborov")

                # --- Block B: Register field and comp star library ---
                _draft_row = pipeline.db.fetch_obs_draft_by_id(int(draft_id))
                if _draft_row is not None:
                    _field_ra = float(_draft_row.get("CENTEROFFIELDRA") or 0.0)
                    _field_dec = float(_draft_row.get("CENTEROFFIELDDE") or 0.0)
                    _arch = Path(str(_draft_row.get("ARCHIVE_PATH") or ""))

                    try:
                        _comp_csv = str(_arch / "platesolve" / "comparison_stars.csv")
                        _var_csv = str(_arch / "platesolve" / "variable_targets.csv")
                        _ms_path = str(_arch / "platesolve" / "MASTERSTAR.fits")

                        _field_id = pipeline.db.register_or_update_field(
                            ra_deg=_field_ra,
                            dec_deg=_field_dec,
                            object_name=_draft_row.get("OBJECT"),
                            masterstar_path=_ms_path,
                            comparison_csv_path=_comp_csv,
                            variable_targets_csv_path=_var_csv,
                            observation_id=str(obs_id),
                        )
                        log_event(
                            f"FIELD_REGISTRY: field_id={_field_id} pre RA={_field_ra:.4f} Dec={_field_dec:.4f}"
                        )
                    except Exception as _fe:  # noqa: BLE001
                        log_event(f"FIELD_REGISTRY zapis zlyhal (nekriticke): {_fe}")
                        _field_id = None

                    if _field_id is not None:
                        try:
                            _comp_path = _arch / "platesolve" / "comparison_stars.csv"
                            if not _comp_path.is_file():
                                raise FileNotFoundError(str(_comp_path))
                            _comp_df = pd.read_csv(
                                _comp_path,
                                low_memory=False,
                                dtype={"catalog_id": str, "name": str},  # Gaia ID musi byt str - float64 straca cifry
                            )

                            _sidecar_rms: dict[str, float] = {}
                            _all_csv: list[Path] = []
                            _sidecar_dir = _arch / "processed"
                            if _sidecar_dir.is_dir():
                                _all_csv = list(_sidecar_dir.rglob("*_catalog.csv"))
                                if _all_csv:
                                    _frames = pd.concat(
                                        [
                                            pd.read_csv(
                                                f,
                                                usecols=lambda c: c in _REGISTRY_COLS,
                                                low_memory=False,
                                                dtype={"catalog_id": str},
                                            )
                                            for f in _all_csv
                                            if Path(f).is_file()
                                        ],
                                        ignore_index=True,
                                    )
                                    if "catalog_id" in _frames.columns and "aperture_mag" in _frames.columns:
                                        _grp = _frames.groupby("catalog_id")["aperture_mag"]
                                        _sidecar_rms = {str(k): float(v) for k, v in _grp.std().items()}

                            _stars_to_upsert: list[dict[str, Any]] = []
                            for _, row in _comp_df.iterrows():
                                cid = str(row.get("catalog_id") or row.get("CATALOG_ID") or "").strip()
                                if not cid:
                                    continue
                                _stars_to_upsert.append(
                                    {
                                        "catalog_id": cid,
                                        "name": str(row.get("name") or "") or None,
                                        "ra_deg": float(row["ra_deg"])
                                        if pd.notna(row.get("ra_deg"))
                                        else None,
                                        "dec_deg": float(row["dec_deg"])
                                        if pd.notna(row.get("dec_deg"))
                                        else None,
                                        "g_mag": float(row["mag"])
                                        if "mag" in row and pd.notna(row.get("mag"))
                                        else None,
                                        "bp_rp": float(row["bp_rp"])
                                        if "bp_rp" in row and pd.notna(row.get("bp_rp"))
                                        else None,
                                        "aperture_median_mag": None,
                                        "aperture_rms": _sidecar_rms.get(cid),
                                        "psf_median_mag": None,
                                        "psf_rms": None,
                                        "n_frames": len(_all_csv) if _all_csv else 0,
                                        "vsx_known_variable": bool(row.get("vsx_known_variable", False)),
                                        "catalog_known_variable": bool(
                                            row.get("catalog_known_variable", False)
                                        ),
                                        "verdict": "Approved",
                                    }
                                )

                            _n_upserted = pipeline.db.upsert_comp_star_library(
                                _field_id,
                                _stars_to_upsert,
                                observation_id=str(obs_id),
                            )
                            log_event(
                                f"COMP_STAR_LIBRARY: {_n_upserted} hviezd ulozenych pre field_id={_field_id}"
                            )

                        except Exception as _ce:  # noqa: BLE001
                            # EXC-0520: T3 -- UI diagnostic/plot only () / except Exception as _ce:  # noqa: BLE001 / log_event(f'COM... (EXCEPT-BULK 2026-07-08)
                            log_event(f"COMP_STAR_LIBRARY zapis zlyhal (nekriticke): {_ce}")

                if approved_by.strip():
                    st.session_state["vyvar_observer_name"] = approved_by.strip()

                st.success(
                    f"[OK] Observation finalized! OBSERVATION ID = {obs_id}. "
                    f"Copied {len(copied)} file(s) to finalized/."
                )
                st.balloons()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Finalization error: {exc}")
                log_event(f"Finalizacia zlyhala: {exc}")


def render_known_field_banner(
    pipeline: Any,
    draft_id: int | None,
) -> None:
    """Show import/calibration banner when the draft field matches ``FIELD_REGISTRY`` (Block C)."""
    if draft_id is None:
        return

    draft_row = pipeline.db.fetch_obs_draft_by_id(int(draft_id))
    if not draft_row:
        return

    ra = float(draft_row.get("CENTEROFFIELDRA") or 0)
    dec = float(draft_row.get("CENTEROFFIELDDE") or 0)
    if ra == 0 and dec == 0:
        return

    from importer import check_known_field

    result = check_known_field(ra, dec, pipeline.db)

    if result is None:
        st.info(
            f"[telescope] New field (RA={ra:.4f} deg, Dec={dec:.4f} deg) - "
            "will be added to the field library after finalization."
        )
        return

    n_obs = result["n_observations"]
    n_comp = result["n_comp_stars"]
    last_obs = result.get("last_observation_id") or "-"

    st.success(
        f"[OK] **Known field!** This field was observed **{n_obs}x**. "
        f"**{n_comp}** verified comparison stars are available "
        f"from prior observations (latest: `{last_obs}`)."
    )

    with st.expander("Show verified comp stars from library", expanded=False):
        if result["comp_stars"]:
            df = pd.DataFrame(result["comp_stars"])
            show_cols = [
                c
                for c in [
                    "NAME",
                    "CATALOG_ID",
                    "G_MAG",
                    "BP_RP",
                    "APERTURE_RMS",
                    "PSF_RMS",
                    "N_OBSERVATIONS",
                    "N_FRAMES_TOTAL",
                    "VERDICT",
                ]
                if c in df.columns
            ]
            st.dataframe(
                df[show_cols] if show_cols else df,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("The library has no approved comp stars for this field.")

    _comp_csv_src = result.get("comparison_csv_path")
    if _comp_csv_src and Path(str(_comp_csv_src)).is_file():
        st.markdown("#### Use existing comp stars")
        st.caption(
            "Instead of a new grid run, you can copy "
            "verified comp stars from a previous observation."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.code(str(_comp_csv_src), language=None)
        with col2:
            if st.button(
                "[clipboard] Use these comp stars",
                key="vyvar_use_known_comp_stars",
                help="Copies comparison_stars.csv into the current draft platesolve/",
            ):
                try:
                    draft_row2 = pipeline.db.fetch_obs_draft_by_id(int(draft_id))
                    _arch = Path(str(draft_row2.get("ARCHIVE_PATH") or ""))
                    _dst = _arch / "platesolve" / "comparison_stars.csv"
                    _dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(_comp_csv_src), _dst)
                    log_event(
                        f"Comp hviezdy skopirovane z kniznice: "
                        f"{_comp_csv_src} -> {_dst}"
                    )
                    st.success(
                        f"[OK] Copied {Path(str(_comp_csv_src)).name} "
                        f"-> current draft platesolve/."
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Copy failed: {exc}")
                    log_event(f"Kopirovanie comp CSV zlyhalo: {exc}")
