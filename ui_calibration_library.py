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
        return "—", "—"
    ie, it = tags.get("id_equipments"), tags.get("id_telescope")
    if ie is None and it is None:
        return "Všeobecné", "Všeobecné"
    cam = tags.get("camera") or tags.get("eq_alias")
    tel = tags.get("telescope") or tags.get("tel_alias")
    return (
        str(cam) if cam else (f"Equipment #{ie}" if ie is not None else "—"),
        str(tel) if tel else (f"Telescope #{it}" if it is not None else "—"),
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
                    "Kamera": cam_lbl,
                    "Ďalekohľad": tel_lbl,
                    "Dátum": date_s,
                    "Vek (dni)": round(age, 2),
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
                    "Kamera": cam_lbl,
                    "Ďalekohľad": tel_lbl,
                    "Dátum": date_s,
                    "Vek (dni)": round(age, 2),
                    "Status": _status_for_age(age, int(flat_limit)),
                    "File": fp.name,
                    "_path": path_res,
                }
            )
    return dark_rows, flat_rows


def _df_for_display(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{k: v for k, v in r.items() if k != "_path"} for r in rows])


def _row_delete_label(row: dict[str, Any], library_root: Path) -> str:
    """Unique label for multiselect (relative path avoids duplicate ``File`` names in subfolders)."""
    p = Path(row["_path"])
    try:
        return str(p.resolve().relative_to(library_root.resolve()))
    except ValueError:
        return p.name


def _render_master_delete_block(
    rows: list[dict[str, Any]],
    *,
    kind_label: str,
    key_prefix: str,
    library_root: Path,
    db: VyvarDatabase | None,
) -> None:
    """Delete any listed masters (age/status irrelevant); updates DB registry when ``db`` is set."""
    if not rows:
        return
    st.caption(
        f"Mazanie súborov ({kind_label}) v priečinku knižnice — **ľubovoľný vek** (nie len expirované)."
    )
    by_label = {_row_delete_label(r, library_root): r["_path"] for r in rows}
    choices = sorted(by_label.keys())
    picked = st.multiselect(
        f"Vyber súbory na zmazanie ({kind_label})",
        options=choices,
        key=f"{key_prefix}_del_pick",
        help="Vyber jeden alebo viac súborov; zmazaním sa odstráni FITS z disku a záznam v CALIBRATION_LIBRARY (ak je DB).",
    )
    if st.button(
        f"Zmazať vybrané ({kind_label})",
        type="secondary",
        key=f"{key_prefix}_del_btn",
    ):
        if not picked:
            st.warning("Nevybral si žiadny súbor.")
        else:
            paths = [Path(by_label[f]) for f in picked if f in by_label]
            n_ok, errs = _delete_paths(paths, db)
            if errs:
                st.error("Časť súborov sa nepodarilo zmazať:\n" + "\n".join(errs))
            st.success(f"Zmazaných súborov: {n_ok}.")
            st.rerun()


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


def render_calibration_library_dashboard(
    *,
    calibration_library_root: Path,
    dark_validity_days: int,
    flat_validity_days: int,
    db: VyvarDatabase | None = None,
) -> None:
    st.subheader("Calibration Library")
    st.caption("Prehľad Master Dark/Flat snímok, veku a validity.")
    st.caption(
        f"Limity: MasterDark = {int(dark_validity_days)} dní, MasterFlat = {int(flat_validity_days)} dní."
    )
    root = Path(calibration_library_root)
    st.caption(f"Knižnica: `{root}`")
    if not root.is_dir():
        st.warning("CalibrationLibrary path neexistuje.")
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

    st.markdown("**Master Darks**")
    if dark_rows:
        ddf = _df_for_display(dark_rows).sort_values(["Status", "Vek (dni)"], ascending=[True, False])
        st.dataframe(ddf.style.applymap(_status_style, subset=["Status"]), use_container_width=True, hide_index=True)
        _render_master_delete_block(
            dark_rows,
            kind_label="Master Dark",
            key_prefix="clib_dark",
            library_root=root,
            db=db,
        )
        exp_d = [r for r in dark_rows if str(r.get("Status")) == "Expired"]
        if exp_d:
            st.caption(f"Skratka — expirované Master Darks: **{len(exp_d)}**")
            if st.button("Vymazať všetky expirované Master Darks", type="secondary", key="clib_del_exp_dark"):
                paths = [Path(r["_path"]) for r in exp_d]
                n_ok, errs = _delete_paths(paths, db)
                if errs:
                    st.error("Časť súborov sa nepodarilo zmazať:\n" + "\n".join(errs))
                st.success(f"Zmazaných súborov: {n_ok}.")
                st.rerun()
    else:
        st.info("V CalibrationLibrary nebol nájdený žiadny Master Dark.")

    st.markdown("**Master Flats**")
    if flat_rows:
        fdf = _df_for_display(flat_rows).sort_values(["Status", "Vek (dni)"], ascending=[True, False])
        st.dataframe(fdf.style.applymap(_status_style, subset=["Status"]), use_container_width=True, hide_index=True)
        _render_master_delete_block(
            flat_rows,
            kind_label="Master Flat",
            key_prefix="clib_flat",
            library_root=root,
            db=db,
        )
        exp_f = [r for r in flat_rows if str(r.get("Status")) == "Expired"]
        if exp_f:
            st.caption(f"Skratka — expirované Master Flats: **{len(exp_f)}**")
            if st.button("Vymazať všetky expirované Master Flats", type="secondary", key="clib_del_exp_flat"):
                paths = [Path(r["_path"]) for r in exp_f]
                n_ok, errs = _delete_paths(paths, db)
                if errs:
                    st.error("Časť súborov sa nepodarilo zmazať:\n" + "\n".join(errs))
                st.success(f"Zmazaných súborov: {n_ok}.")
                st.rerun()
    else:
        st.info("V CalibrationLibrary nebol nájdený žiadny Master Flat.")

    st.markdown("---")
    st.markdown("**Generovanie masterov do knižnice**")
    st.caption(
        "Zadaj adresár so **surovými** dark alebo flat FITS (vrátane podpriečinkov). "
        "Skladanie a názov súboru (napr. `Dark_120s_Dark_0G_-10deg_Bin2_YYYYMMDD.fits`, `Flat_…NoFilter…`) "
        "je rovnaké ako pri importe — z hlavičiek EXPTIME, FILTER, GAIN, teplota CCD, binning, DATE-OBS."
    )
    st.caption(
        "**Povinný set:** pred generovaním musíš vybrať **kameru (Equipment)** aj **ďalekohľad (Telescope)** — "
        "master sa zaregistruje v knižnici pod týmto setom."
    )
    gen_eq_id: int | None = None
    gen_tel_id: int | None = None
    gen_set_ok = False
    if db is None:
        st.warning("Databáza nie je k dispozícii — generovanie s zápisom setu do knižnice nie je možné.")
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
                "V databáze musí existovať aspoň jedna **aktívna kamera** a jeden **aktívny ďalekohľad**. "
                "Bez toho nie je možné vybrať set pre zápis mastera."
            )
        else:
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                glab_eq = st.selectbox(
                    "Equipment (povinné)",
                    options=gel,
                    key="clib_gen_equipment",
                )
            with gcol2:
                glab_tel = st.selectbox(
                    "Telescope (povinné)",
                    options=gtl,
                    key="clib_gen_telescope",
                )
            gen_eq_id = int(gen_eq_opts[glab_eq])
            gen_tel_id = int(gen_tel_opts[glab_tel])
            gen_set_ok = True
    dark_src = st.text_input(
        "Cesta k surovým dark snímkam",
        value="",
        key="clib_gen_dark_src",
        help="Adresár, ktorý obsahuje dark FITS (IMAGETYP alebo názov).",
    )
    flat_src = st.text_input(
        "Cesta k surovým flat snímkam",
        value="",
        key="clib_gen_flat_src",
        help="Adresár so flat FITS.",
    )
    gc1, gc2 = st.columns(2)
    with gc1:
        if st.button(
            "Generuj Master Dark",
            type="primary",
            key="clib_btn_gen_dark",
            disabled=not gen_set_ok,
        ):
            p = Path(dark_src.strip())
            if not gen_set_ok or gen_eq_id is None or gen_tel_id is None:
                st.error("Vyber kameru aj ďalekohľad (set).")
            elif not str(dark_src).strip():
                st.warning("Zadaj cestu k darkom.")
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
                        if m.startswith("✅"):
                            st.success(m)
                        elif m.startswith("ℹ️"):
                            st.info(m)
                        else:
                            st.error(m)
                    if out is not None and all(not str(x).startswith("❌") for x in msgs):
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Chyba: {exc}")
    with gc2:
        if st.button(
            "Generuj Master Flat",
            type="primary",
            key="clib_btn_gen_flat",
            disabled=not gen_set_ok,
        ):
            p = Path(flat_src.strip())
            if not gen_set_ok or gen_eq_id is None or gen_tel_id is None:
                st.error("Vyber kameru aj ďalekohľad (set).")
            elif not str(flat_src).strip():
                st.warning("Zadaj cestu k flatom.")
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
                        if m.startswith("✅"):
                            st.success(m)
                        elif m.startswith("ℹ️"):
                            st.info(m)
                        else:
                            st.error(m)
                    if out is not None and all(not str(x).startswith("❌") for x in msgs):
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Chyba: {exc}")
