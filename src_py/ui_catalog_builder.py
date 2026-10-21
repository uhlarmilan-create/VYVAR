"""Catalog builder in Settings -> Paths and catalogs (CATALOG-WIZARD-2026-10, Milan 2026-10-20).

Buttons for the scripts in ``catalogs/``; each runs as its own process (``catalog_jobs``), so a
build of hours survives Streamlit reruns. Milan's decisions 2026-10-20:

- VSX + exoplanets: "Update" replaces both tables completely (staging table, swap only when the
  whole download arrived; added / removed / changed rows are reported).
- Gaia DR3: build a missing DB, resume an unfinished build, and maintenance (report, register,
  drop unused indexes). Compacting a 38 GB DB stays a command-line step (gaia_maintain.py
  --vacuum-into). Gaia DR3 does not change; DR4 (2 December 2026) will be a new file.
- Blind index: build / complete it from the Gaia DB.

The jobs write to the paths saved in config.json; a path edited but not saved must be saved first.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import streamlit as st

import catalog_jobs as cj
import settings_model as sm

_K = "vyvar_catb_"

#: full-sky Gaia DR3 stars (millions) with G <= limit, counted in Milan's DB (2026-10-20, COUNT on
#: idx_g_mag); size per star from the same DB (38.0 GB for 211.7 M stars with the 3 indexes)
_GAIA_STARS_M = {12.0: 3.05, 13.0: 7.27, 14.0: 16.53, 15.0: 36.01, 16.0: 75.48, 16.5: 107.62, 17.0: 151.74, 17.5: 211.71}
_GB_PER_M = 38.0 / 211.71


def _gaia_estimate(g: float) -> str:
    keys = sorted(_GAIA_STARS_M)
    if g > keys[-1]:
        return f"more than {_GAIA_STARS_M[keys[-1]]:.0f} million stars and {_GAIA_STARS_M[keys[-1]] * _GB_PER_M:.0f} GB"
    g = max(g, keys[0])
    lo = max(k for k in keys if k <= g)
    hi = min(k for k in keys if k >= g)
    n = _GAIA_STARS_M[lo] if hi == lo else _GAIA_STARS_M[lo] + (_GAIA_STARS_M[hi] - _GAIA_STARS_M[lo]) * (g - lo) / (hi - lo)
    return f"about {n:.0f} million stars, {n * _GB_PER_M:.0f} GB"


def _free_gb(path: Path) -> float | None:
    for p in (path, *path.parents):
        if p.exists():
            try:
                return shutil.disk_usage(p).free / 1024 ** 3
            except OSError:
                return None
    return None


def _resolved(raw: Any, default: str = "") -> Path | None:
    from catalog_db import resolve_catalog_path  # noqa: PLC0415

    s = str(raw or "").strip() or default
    return resolve_catalog_path(s) if s else None


def _job_panel(data_root: Path, catalog: str) -> bool:
    """Latest job of ``catalog``: progress, log tail, Stop. Returns True while one is running."""
    jd = cj.latest_job(data_root, catalog)
    if jd is None:
        return False
    running = cj.job_state(jd) == "running"

    def _body() -> None:
        state = cj.job_state(jd)
        info = cj.read_status(jd) or {}
        label = str(info.get("label") or jd.name)
        if state == "running":
            frac = cj.progress(jd)
            st.progress(frac if frac is not None else 0.0,
                        text=f"{label}: running" + (f" - {frac:.0%}" if frac is not None else ""))
            if st.button("Stop", key=f"{_K}stop_{jd.name}"):
                cj.stop(jd)
                st.rerun(scope="app")
        else:
            fn = {"done": st.success, "failed": st.error}.get(state, st.warning)
            fn(f"{label}: {state} ({str(info.get('finished_utc') or info.get('updated_utc') or '')[:16]} UTC) - "
               f"{str(info.get('message') or '')[:300]}")
        with st.expander("Log", expanded=state != "done"):
            st.code("\n".join(cj.tail(jd, 15)) or "(empty)", language=None)
            st.caption(str(jd))
        seen = st.session_state.setdefault(f"{_K}seen_running", set())
        if state == "running":
            seen.add(jd.name)
        elif jd.name in seen:  # finished while the page was open: refresh the catalog cards once
            seen.discard(jd.name)
            st.rerun(scope="app")

    if running:
        st.fragment(run_every=3.0)(_body)()
    else:
        _body()
    return running


def _start(tool: str, args: list[str], data_root: Path) -> None:
    try:
        cj.launch(tool, args, data_root)
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(str(exc))
        return
    st.rerun(scope="app")


def _gaia(data_root: Path, current: dict[str, Any], locked: bool) -> None:
    from catalog_db import GAIA_DB_DEFAULT  # noqa: PLC0415

    target = _resolved(current.get("gaia_db_path"), GAIA_DB_DEFAULT)
    status = sm.gaia_status(str(target)) if target and target.is_file() else None
    st.caption("Gaia DR3 does not change; Gaia DR4 (2 December 2026) will be a new catalog file.")
    busy = _job_panel(data_root, "gaia")
    disabled = busy or locked
    if status is None:
        st.markdown(f"**Build** a new catalog: `{target}`")
        c1, c2, c3 = st.columns(3)
        g = c1.number_input("G limit", 12.0, 18.0, 17.5, 0.5, key=f"{_K}g_mag")
        d0 = c2.number_input("Dec from", -90.0, 89.0, -90.0, 1.0, key=f"{_K}g_d0")
        d1 = c3.number_input("Dec to", -89.0, 90.0, 90.0, 1.0, key=f"{_K}g_d1")
        free = _free_gb(target)
        st.caption(f"Full sky at G <= {g:g}: {_gaia_estimate(g)} (less for a Dec band); ESA archive download, "
                   f"hours to days, resumable. Free on the target drive: "
                   + (f"{free:.0f} GB" if free is not None else "?") + ". "
                   "VYVAR fetches fainter stars of a field online when a frame needs them.")
        if st.button("Build Gaia catalog", type="primary", disabled=disabled or d1 <= d0, key=f"{_K}g_build"):
            _start("gaia_build", cj.tool_args("gaia_build", out=target, mag_limit=g, dec_min=d0, dec_max=d1), data_root)
        return
    if status.state == "unfinished":
        last = cj.latest_job(data_root, "gaia", "gaia_build")
        args = cj.job_args(last) if last else []
        if args:
            st.markdown(f"**Resume** the unfinished build with the same options: `{' '.join(args)}`")
            if st.button("Resume build", type="primary", disabled=disabled, key=f"{_K}g_resume"):
                _start("gaia_build", args, data_root)
        else:
            st.info("The build was started outside VYVAR: run the same command again to resume it "
                    "(python catalogs/build_gaia_catalog.py with the same options).")
        return
    st.markdown("**Maintenance** (close running VYVAR runs first: register and drop need the DB alone)")
    b1, b2, b3 = st.columns(3)
    if b1.button("Report", disabled=disabled, key=f"{_K}g_report"):
        _start("gaia_report", cj.tool_args("gaia_report", db=target), data_root)
    if b2.button("Register (catalog_meta)", disabled=disabled, key=f"{_K}g_register",
                 help="Counts the stars once (about a minute) and writes the catalog record."):
        _start("gaia_register", cj.tool_args("gaia_register", db=target), data_root)
    extra = "Unused index" in (status.note or "")
    if b3.button("Drop unused indexes", disabled=disabled or not extra, key=f"{_K}g_drop"):
        _start("gaia_drop_unused", cj.tool_args("gaia_drop_unused", db=target), data_root)
    st.caption("Compacting after dropping indexes needs a full copy (38 GB, minutes): "
               "python catalogs/gaia_maintain.py --vacuum-into catalogs/vyvar_gaia_dr3_new.db")


def _objects(data_root: Path, current: dict[str, Any], locked: bool) -> None:
    from catalog_db import OBJECTS_DB_DEFAULT, read_catalog_meta  # noqa: PLC0415

    target = _resolved(current.get("objects_db_path"), OBJECTS_DB_DEFAULT)
    meta = read_catalog_meta(target) if target and target.is_file() else {}
    vp = (meta.get("vsx_data") or {}).get("params") or {}
    busy = _job_panel(data_root, "objects")
    st.markdown(f"**{'Update' if meta else 'Build'}** `{target}`: downloads the complete tables and replaces "
                "them only when everything arrived (the old tables stay on any error). VSX: one VizieR "
                "query per degree of declination (180 for the whole sky); exoplanets: 2 NASA archive queries.")
    c1, c2, c3, c4 = st.columns(4)
    do_vsx = c1.checkbox("VSX", True, key=f"{_K}o_vsx")
    do_exo = c2.checkbox("Exoplanets", True, key=f"{_K}o_exo")
    mag = c3.number_input("VSX: max brighter than", 10.0, 25.0, float(vp.get("mag_limit_max") or 18.0), 0.5,
                          key=f"{_K}o_mag", disabled=not do_vsx)
    c4.caption(f"Dec {vp.get('dec_min', -90.0):g} .. {vp.get('dec_max', 90.0):g}")
    if st.button("Update VSX + exoplanets" if meta else "Build objects catalog", type="primary",
                 disabled=busy or locked or not (do_vsx or do_exo), key=f"{_K}o_go"):
        _start("objects_update", cj.tool_args(
            "objects_update", db=target, vsx=do_vsx, exoplanets=do_exo, vsx_mag_limit=mag,
            vsx_dec_min=float(vp.get("dec_min", -90.0)), vsx_dec_max=float(vp.get("dec_max", 90.0))), data_root)


def _index(data_root: Path, current: dict[str, Any], locked: bool) -> None:
    from astrosolve import default_index_dir  # noqa: PLC0415
    from catalog_db import GAIA_DB_DEFAULT  # noqa: PLC0415

    gaia = _resolved(current.get("gaia_db_path"), GAIA_DB_DEFAULT)
    idx_raw = _resolved(current.get("astrometry_index_dir"))
    out = Path(default_index_dir(str(gaia), str(idx_raw) if idx_raw else None))
    status = sm.blind_index_status(current.get("astrometry_index_dir"), str(gaia))
    busy = _job_panel(data_root, "astro_index")
    gaia_ok = gaia is not None and gaia.is_file()
    if status.state == "ready":
        st.caption(f"Complete: `{out}`. To rebuild after a new Gaia DB, rename the folder and build again.")
        return
    st.markdown(f"**{'Complete' if status.state == 'unfinished' else 'Build'}** the blind plate-solve index: `{out}`")
    free = _free_gb(out)
    st.caption("Only frames without a position in the FITS header need it. About 7 GB from a full-sky Gaia DB; "
               "one pass over the DB, then 9 pattern scales; finished scales are kept (resumable). Free: "
               + (f"{free:.0f} GB" if free is not None else "?"))
    if st.button("Build blind index", type="primary", disabled=busy or locked or not gaia_ok, key=f"{_K}i_go"):
        _start("astro_index", cj.tool_args("astro_index", db=gaia, out=out), data_root)


def render_catalog_builder(cfg: Any, current: dict[str, Any], draft: dict[str, Any]) -> None:
    """Section under the catalog cards of Settings -> Paths and catalogs."""
    data_root = Path(cfg.data_root)
    keys = ("gaia_db_path", "objects_db_path", "astrometry_index_dir")
    locked = any(str(draft.get(k) or "") != str(current.get(k) or "") for k in keys)
    st.markdown("##### Catalog builder")
    if locked:
        st.info("A catalog path was edited: Save it first, then build (the builders write to the saved paths).")
    t1, t2, t3 = st.tabs(["Gaia DR3", "VSX + exoplanets", "Blind index"])
    with t1:
        _gaia(data_root, current, locked)
    with t2:
        _objects(data_root, current, locked)
    with t3:
        _index(data_root, current, locked)
