"""UI: final approval step — persist OBSERVATION and archive key artifacts under ``finalized/``."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from astropy.io import fits

from infolog import log_event


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
                    log_event(f"Finalizácia: preskočený chýbajúci súbor {src}")
            except Exception as exc:  # noqa: BLE001
                log_event(f"Finalizácia: kopírovanie zlyhalo {src} → {dst}: {exc!s}")

        pcsv = root / "processed"
        if pcsv.is_dir():
            out_flat = fin / "per_frame_csv"
            try:
                out_flat.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                log_event(f"Finalizácia: nemôžem vytvoriť {out_flat}: {exc!s}")
            else:
                for src in pcsv.rglob("*_catalog.csv"):
                    try:
                        if src.is_file():
                            dst = out_flat / src.name
                            shutil.copy2(src, dst)
                            copied.append(f"per_frame_csv/{dst.name}")
                    except Exception as exc:  # noqa: BLE001
                        log_event(f"Finalizácia: kopírovanie zlyhalo {src}: {exc!s}")
    except Exception as exc:  # noqa: BLE001
        log_event(f"Finalizácia: _copy_finalization_files: {exc!s}")
    return copied


def _draft_scan_row(db: Any, draft_id: int) -> dict[str, Any] | None:
    try:
        row = db.conn.execute(
            """
            SELECT s.EXPTIME AS exptime, s.FILTERS AS filters, s.BINNING AS binning
            FROM OBS_DRAFT d
            LEFT JOIN SCANNING s ON s.ID = d.ID_SCANNING
            WHERE d.ID = ?;
            """,
            (int(draft_id),),
        ).fetchone()
        return dict(row) if row is not None else None
    except Exception:  # noqa: BLE001
        return None


def _draft_location_name(db: Any, draft_id: int) -> str:
    try:
        row = db.conn.execute(
            """
            SELECT l.PLACENAME AS place_name
            FROM OBS_DRAFT d
            LEFT JOIN LOCATION l ON l.ID = d.ID_LOCATION
            WHERE d.ID = ?;
            """,
            (int(draft_id),),
        ).fetchone()
        if row is None:
            return "—"
        v = row["place_name"] if hasattr(row, "keys") else row[0]
        s = str(v).strip() if v is not None else ""
        return s or "—"
    except Exception:  # noqa: BLE001
        return "—"


def _n_light_frames(db: Any, draft_id: int) -> int:
    try:
        row = db.conn.execute(
            """
            SELECT COUNT(*) AS n FROM OBS_FILES
            WHERE DRAFT_ID = ? AND LOWER(COALESCE(IMAGETYP, '')) = 'light';
            """,
            (int(draft_id),),
        ).fetchone()
        if row is None:
            return 0
        return int(row["n"] if hasattr(row, "keys") else row[0])
    except Exception:  # noqa: BLE001
        return 0


def _csv_nonempty(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        df = pd.read_csv(path, nrows=5000)
        return len(df) > 0
    except Exception:  # noqa: BLE001
        return False


def render_finalization(
    pipeline: Any,
    draft_id: int | None,
) -> None:
    st.subheader("Finalizácia pozorovania")
    st.caption(
        "Skontroluj všetky záložky pred schválením. Po schválení sa pozorovanie zapíše natrvalo."
    )

    if draft_id is None:
        st.warning("Vyber alebo spracuj draft (žiadne vybrané pozorovanie).")
        return

    db = pipeline.db
    row = db.fetch_obs_draft_by_id(int(draft_id))
    if row is None:
        st.error(f"Draft {draft_id} sa nenašiel v databáze.")
        return

    arch_raw = row.get("ARCHIVE_PATH")
    try:
        archive_path = Path(str(arch_raw)).expanduser().resolve() if arch_raw else None
    except OSError:
        archive_path = None

    st.markdown("#### Kontrolný zoznam")
    checks: list[tuple[str, bool]] = []

    ms_path = (archive_path / "platesolve" / "MASTERSTAR.fits") if archive_path else None
    checks.append(("MASTERSTAR existuje", bool(ms_path and ms_path.is_file())))

    plate_ok = False
    if ms_path and ms_path.is_file():
        try:
            with fits.open(ms_path, memmap=False) as hdul:
                h0 = hdul[0].header
                plate_ok = ("VY_SIPRF" in h0) or ("VY_PSOLV" in h0)
        except Exception:  # noqa: BLE001
            plate_ok = False
    checks.append(("Plate solve OK (VY_SIPRF alebo VY_PSOLV v MASTERSTAR)", plate_ok))

    n_cat = 0
    if archive_path and (archive_path / "processed").is_dir():
        n_cat = sum(1 for _ in (archive_path / "processed").rglob("*_catalog.csv"))
    checks.append(("Per-frame CSV existuje (aspoň 1 × *_catalog.csv v processed/)", n_cat >= 1))

    comp_path = (archive_path / "platesolve" / "comparison_stars.csv") if archive_path else None
    checks.append(
        ("Comp stars definované (comparison_stars.csv, >0 riadkov)", _csv_nonempty(comp_path) if comp_path else False)
    )

    var_path = (archive_path / "platesolve" / "variable_targets.csv") if archive_path else None
    checks.append(
        (
            "Variable targets definované (variable_targets.csv, >0 riadkov)",
            _csv_nonempty(var_path) if var_path else False,
        )
    )

    st_fin = str(row.get("STATUS") or "").strip().upper()
    checks.append(("Draft nie je už finalizovaný", st_fin != "FINALIZED"))

    for label, ok in checks:
        st.markdown(f"{'✅' if ok else '⚠️'} {label}")

    if not all(c[1] for c in checks):
        st.warning("Niektoré kontroly zlyhali — môžeš aj tak pokračovať (rozhodnutie je na tebe).")

    st.markdown("#### Súhrn pozorovania")
    scan = _draft_scan_row(db, int(draft_id))
    loc_name = _draft_location_name(db, int(draft_id))
    tel_eq = db.fetch_obs_draft_telescope_equipment(int(draft_id)) or {}
    n_frames = _n_light_frames(db, int(draft_id))

    ra_v = row.get("CENTEROFFIELDRA")
    de_v = row.get("CENTEROFFIELDDE")
    ra_s = f"{float(ra_v):.6f}°" if ra_v is not None else "—"
    de_s = f"{float(de_v):.6f}°" if de_v is not None else "—"
    obj = str(row.get("OBJECT") or "").strip() or "—"
    flt = str((scan or {}).get("filters") or "").strip() or "—"
    expt = (scan or {}).get("exptime")
    expt_s = f"{float(expt):.2f} s" if expt is not None else "—"
    binv = (scan or {}).get("binning")
    bin_s = str(int(binv)) if binv is not None else "—"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Objekt:** {obj}")
        st.markdown(f"**RA / Dec:** {ra_s} / {de_s}")
        st.markdown(f"**Filter:** {flt}")
        st.markdown(f"**Expozícia:** {expt_s}")
        st.markdown(f"**Binning:** {bin_s}")
        st.markdown(f"**Počet light snímok (OBS_FILES):** {n_frames}")
    with c2:
        st.markdown(f"**DATE_OBS start:** {row.get('DATE_OBS_START') or '—'}")
        st.markdown(f"**DATE_OBS end:** {row.get('DATE_OBS_END') or '—'}")
        st.markdown(f"**Equipment:** {tel_eq.get('equipment_name') or '—'}")
        st.markdown(f"**Telescope:** {tel_eq.get('telescope_name') or '—'}")
        st.markdown(f"**Location:** {loc_name}")
        st.markdown(f"**Archive path:** `{archive_path or '—'}`")

    st.markdown("#### Schválenie")
    approved_by = st.text_input(
        "Meno pozorovateľa (pre zápis do OBSERVATION)",
        value=st.session_state.get("vyvar_observer_name", ""),
        key="vyvar_finalization_approved_by",
    )
    notes = st.text_area(
        "Poznámky k pozorovaniu (voliteľné)",
        key="vyvar_finalization_notes",
        height=80,
    )

    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(
            "Po schválení sa:\n"
            "1. Pozorovanie zapíše do tabuľky OBSERVATION\n"
            "2. Dôležité súbory sa skopírujú do finalized/\n"
            "3. Draft zostane v DB (manuálny cleanup v Database Explorer)"
        )
    with col2:
        confirm = st.checkbox(
            "Rozumiem, chcem finalizovať",
            key="vyvar_finalization_confirm",
        )
        finalize_btn = st.button(
            "✅ Schváliť a finalizovať",
            type="primary",
            disabled=not confirm,
            key="vyvar_finalization_go",
        )

    if finalize_btn and confirm:
        with st.spinner("Finalizujem pozorovanie..."):
            try:
                obs_id = pipeline.db.finalize_draft_to_observation(
                    int(draft_id),
                    approved_by=approved_by.strip() or None,
                    notes=notes.strip() or None,
                )
                log_event(f"Finalizácia: draft {draft_id} → OBSERVATION {obs_id}")

                if archive_path is None:
                    st.warning("ARCHIVE_PATH nie je nastavený — preskakujem kopírovanie súborov.")
                    copied: list[str] = []
                else:
                    copied = _copy_finalization_files(archive_path, int(draft_id), obs_id)
                log_event(f"Finalizácia: skopírovaných {len(copied)} súborov")

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
                        log_event(f"FIELD_REGISTRY zápis zlyhal (nekritické): {_fe}")
                        _field_id = None

                    if _field_id is not None:
                        try:
                            _comp_path = _arch / "platesolve" / "comparison_stars.csv"
                            if not _comp_path.is_file():
                                raise FileNotFoundError(str(_comp_path))
                            _comp_df = pd.read_csv(_comp_path)

                            _sidecar_rms: dict[str, float] = {}
                            _all_csv: list[Path] = []
                            _sidecar_dir = _arch / "processed"
                            if _sidecar_dir.is_dir():
                                _all_csv = list(_sidecar_dir.rglob("*_catalog.csv"))
                                if _all_csv:
                                    _frames = pd.concat(
                                        [pd.read_csv(f) for f in _all_csv],
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
                                f"COMP_STAR_LIBRARY: {_n_upserted} hviezd uložených pre field_id={_field_id}"
                            )

                        except Exception as _ce:  # noqa: BLE001
                            log_event(f"COMP_STAR_LIBRARY zápis zlyhal (nekritické): {_ce}")

                if approved_by.strip():
                    st.session_state["vyvar_observer_name"] = approved_by.strip()

                st.success(
                    f"✅ Pozorovanie finalizované! OBSERVATION ID = {obs_id}. "
                    f"Skopírovaných {len(copied)} súborov do finalized/."
                )
                st.balloons()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Chyba pri finalizácii: {exc}")
                log_event(f"Finalizácia zlyhala: {exc}")


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
            f"🔭 Nové pole (RA={ra:.4f}°, Dec={dec:.4f}°) — "
            "po finalizácii sa pridá do knižnice polí."
        )
        return

    field = result["field"]
    n_obs = result["n_observations"]
    n_comp = result["n_comp_stars"]
    last_obs = result.get("last_observation_id") or "—"

    st.success(
        f"✅ **Známe pole!** Toto pole bolo pozorované **{n_obs}×**. "
        f"K dispozícii je **{n_comp}** overených porovnávacích hviezd "
        f"z predchádzajúcich pozorovaní (posledné: `{last_obs}`)."
    )

    with st.expander("Zobraziť overené comp hviezdy z knižnice", expanded=False):
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
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Knižnica neobsahuje schválené comp hviezdy pre toto pole.")

    _comp_csv_src = result.get("comparison_csv_path")
    if _comp_csv_src and Path(str(_comp_csv_src)).is_file():
        st.markdown("#### Použiť existujúce comp hviezdy")
        st.caption(
            "Namiesto nového grid výpočtu môžeš skopírovať "
            "overené comp hviezdy z predchádzajúceho pozorovania."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.code(str(_comp_csv_src), language=None)
        with col2:
            if st.button(
                "📋 Použiť tieto comp hviezdy",
                key="vyvar_use_known_comp_stars",
                help="Skopíruje comparison_stars.csv do platesolve/ aktuálneho draftu",
            ):
                try:
                    draft_row2 = pipeline.db.fetch_obs_draft_by_id(int(draft_id))
                    _arch = Path(str(draft_row2.get("ARCHIVE_PATH") or ""))
                    _dst = _arch / "platesolve" / "comparison_stars.csv"
                    _dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(_comp_csv_src), _dst)
                    log_event(
                        f"Comp hviezdy skopírované z knižnice: "
                        f"{_comp_csv_src} → {_dst}"
                    )
                    st.success(
                        f"✅ Skopírované {Path(str(_comp_csv_src)).name} "
                        f"→ platesolve/ aktuálneho draftu."
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Kopírovanie zlyhalo: {exc}")
                    log_event(f"Kopírovanie comp CSV zlyhalo: {exc}")
