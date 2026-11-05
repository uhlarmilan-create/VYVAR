"""VYVAR v12 rig (2026-11-05): auto-detect really sets the selectors (RIG-AUTOSELECT-2026-11, site included,
supersedes S6-FIX-SITE-MANUAL-2026-10; SITE-MATCH-2026-11 tolerance), the run is blocked when the selection
contradicts the light headers (RIG-GUARD-2026-11, GUI and CLI one path) and a rig / site of the header can be
added to the DB on explicit confirmation (RIG-ADD-2026-11)."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
APP_SCRIPT = f"import sys\nsys.path.insert(0, {str(SRC)!r})\nimport app\napp.main()\n"

C5A = {"ID": 1, "CAMERANAME": "C5A-150M", "ALIAS": "c5a", "SENSORTYPE": "CMOS", "SENSORSIZE": "14208*10656",
       "PIXELSIZE": 3.76, "GAIN_ADU": None, "MODEL_KEY": None}
QHY = {"ID": 2, "CAMERANAME": "QHY294MM", "ALIAS": "qhy", "SENSORTYPE": "CMOS", "SENSORSIZE": "4164*2796",
       "PIXELSIZE": 4.63, "GAIN_ADU": None, "MODEL_KEY": None}
AZ800 = {"ID": 1, "TELESCOPENAME": "AZ800", "ALIAS": "az", "DIAMETER": 800.0, "FOCAL": 5480.0}
ZEISS = {"ID": 2, "TELESCOPENAME": "Carl-Zeiss 200mm", "ALIAS": "cz", "DIAMETER": 200.0, "FOCAL": 1000.0}
JIRNY = {"id": 1, "name": "Jirny", "lat": 50.1120, "lon": 14.6980, "alt_m": 275.0}
DABLICE = {"id": 2, "name": "Dablice", "lat": 50.1310, "lon": 14.4870, "alt_m": 280.0}


def _hdr(**kw):
    from astropy.io import fits

    return fits.Header(kw)


def _c5a_az800_header(**extra):
    kw = {"IMAGETYP": "LIGHT", "INSTRUME": "C5A-150M", "NAXIS1": 3552, "NAXIS2": 2664, "XBINNING": 4,
          "YBINNING": 4, "FOCALLEN": 5480.0, "APTDIA": 800.0, "TELESCOP": "AZ800"}
    kw.update(extra)
    return _hdr(**kw)


def _write_c5a_light(path: Path, **extra) -> None:
    """A C5A-150M / AZ800 light frame on disk (16x binned so the file stays small; NAXIS from the data)."""
    from astropy.io import fits

    h = _c5a_az800_header(**extra)
    for k in ("NAXIS1", "NAXIS2"):
        del h[k]
    h["XBINNING"], h["YBINNING"] = 16, 16
    fits.PrimaryHDU(np.zeros((666, 888), dtype=np.uint8), header=h).writeto(path)


def _src_app():
    mod = sys.modules.get("app")
    if mod is not None and Path(getattr(mod, "__file__", "") or "").resolve() != (SRC / "app.py").resolve():
        del sys.modules["app"]
    if str(SRC) in sys.path:
        sys.path.remove(str(SRC))
    sys.path.insert(0, str(SRC))
    import app

    return app


# ----------------------------------------------------------------- A. auto-detect sets the selector widgets
def test_autodetect_assigns_the_widget_keys_by_band(monkeypatch) -> None:
    """Root cause: a keyed selectbox is identified by its key only, so the old ``index=`` + pop never reached
    the browser. The widget keys themselves are now assigned: high -> set, medium -> set + unconfirmed,
    low / none -> kept."""
    from optics_autodetect import AutodetectReport, Detection, SiteDetection

    app = _src_app()
    state: dict = {"vyvar_varstrem_equipment": "2: QHY294MM (qhy)", "vyvar_varstrem_telescope": "2: Carl-Zeiss",
                   "vyvar_varstrem_location": "2: Dablice"}
    monkeypatch.setattr(app.st, "session_state", state)
    rep = AutodetectReport(
        equipment=Detection(1, "1: C5A-150M (c5a)", 0.90, ["name"]),
        telescope=Detection(1, "1: AZ800 (az)", 0.60, ["FOCALLEN"]),
        location=SiteDetection(header_site=(50.112, 14.698, None), matched_id=1, label="1: Jirny", n_within=1),
        header_path="x.fits",
    )
    labels = dict(eq_labels=["1: C5A-150M (c5a)", "2: QHY294MM (qhy)"],
                  tel_labels=["1: AZ800 (az)", "2: Carl-Zeiss"], loc_labels=["1: Jirny", "2: Dablice"])
    app._rig_apply_autodetect(rep, **labels)
    assert state["vyvar_varstrem_equipment"] == state["vyvar_rig_equipment"] == "1: C5A-150M (c5a)"
    assert state["vyvar_varstrem_telescope"] == "1: AZ800 (az)"
    assert state["vyvar_varstrem_location"] == state["vyvar_rig_location"] == "1: Jirny"
    assert set(state["vyvar_rig_unconfirmed"]) == {"vyvar_rig_telescope"}
    # low / none: the current selection stays
    low = AutodetectReport(equipment=Detection(1, "1: C5A-150M (c5a)", 0.40, []), telescope=Detection(),
                           location=SiteDetection(reasons=["no site in the header"]), header_path="x.fits")
    state.update({"vyvar_varstrem_equipment": "2: QHY294MM (qhy)", "vyvar_varstrem_location": "2: Dablice"})
    app._rig_apply_autodetect(low, **labels)
    assert state["vyvar_varstrem_equipment"] == "2: QHY294MM (qhy)"
    assert state["vyvar_varstrem_location"] == "2: Dablice" and state["vyvar_rig_unconfirmed"] == {}


def test_rig_selectboxes_carry_the_value_by_key_not_index() -> None:
    """No ``index=`` next to the key (Streamlit's default / session-state conflict), and the auto-detect button
    applies its result in an on_click callback, i.e. before the selectors render."""
    tree = ast.parse((SRC / "app.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_live_view")
    rig_boxes = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "selectbox":
            key = next((k.value for k in node.keywords if k.arg == "key"), None)
            if key is not None and "_RIG_WIDGET" in ast.unparse(key):
                rig_boxes.append(node)
                assert "index" not in {k.arg for k in node.keywords}, ast.unparse(node)
    assert len(rig_boxes) == 3
    btn = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "button"
           and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "Auto-detect optics from FITS"]
    assert len(btn) == 1 and ast.unparse(next(k.value for k in btn[0].keywords if k.arg == "on_click")) == \
        "_rig_autodetect_cb"
    assert "[test] Auto-detect" not in (SRC / "app.py").read_text(encoding="utf-8")


def test_widget_prepare_restores_the_selection_after_a_view_switch(monkeypatch) -> None:
    app = _src_app()
    state: dict = {"vyvar_rig_equipment": "2: B"}
    monkeypatch.setattr(app.st, "session_state", state)
    app._rig_widget_prepare("vyvar_rig_equipment", ["1: A", "2: B"])  # widget state dropped by the view switch
    assert state["vyvar_varstrem_equipment"] == "2: B"
    state["vyvar_varstrem_equipment"] = "9: gone"  # a hidden / deleted row
    app._rig_widget_prepare("vyvar_rig_equipment", ["1: A", "2: B"])
    assert state["vyvar_varstrem_equipment"] == "2: B"
    state["vyvar_varstrem_equipment"] = "1: A"  # the user's choice is kept
    app._rig_widget_prepare("vyvar_rig_equipment", ["1: A", "2: B"])
    assert state["vyvar_varstrem_equipment"] == "1: A"


# ------------------------------------------------------------------ A. site detection + derived tolerance
def test_site_tolerance_is_derived_from_the_header_precision() -> None:
    from param_resolver import header_site_reading

    import optics_autodetect as oa

    fine = _hdr(SITELAT=50.112345, SITELONG=14.698765)
    coarse = _hdr(SITELAT=50.1, SITELONG=14.7)
    t_fine = oa.site_match_tolerance_m(header_site_reading(fine))
    t_coarse = oa.site_match_tolerance_m(header_site_reading(coarse))
    assert oa.SITE_SAME_RADIUS_M < t_fine < oa.SITE_SAME_RADIUS_M + 1.0
    # 0.05 deg rounding: 5.56 km in latitude, 3.57 km in longitude at 50 deg
    assert t_coarse == pytest.approx(oa.SITE_SAME_RADIUS_M + np.hypot(5559.7, 3573.4), rel=2e-3)
    # the card text decides, not the float: "50.10000" states 5 decimals
    from astropy.io import fits

    h = fits.Header.fromstring("SITELAT =             50.10000 / deg".ljust(80)
                               + "SITELONG=             14.70000 / deg".ljust(80) + "END".ljust(80))
    assert oa.site_match_tolerance_m(header_site_reading(h)) < oa.SITE_SAME_RADIUS_M + 1.0


def test_detect_location_matches_nearest_site_within_tolerance() -> None:
    import optics_autodetect as oa

    det = oa.detect_location(_hdr(SITELAT=50.1121, SITELONG=14.6982, SITEELEV=275.0), [JIRNY, DABLICE])
    assert det.matched_id == 1 and det.label == "1: Jirny" and det.band() == "high" and det.distance_m < 50
    # OBSGEO-B/L (the resolver's keyword family) work the same way
    assert oa.detect_location(_hdr(**{"OBSGEO-B": 50.131, "OBSGEO-L": 14.487}), [JIRNY, DABLICE]).matched_id == 2
    # a site 6 km from Jirny is no DB site: selector kept, addable
    far = oa.detect_location(_hdr(SITELAT=50.112, SITELONG=14.782), [JIRNY, DABLICE])
    assert far.matched_id is None and far.addable and "selector kept" in far.reasons[0]
    # two DB sites inside a coarse header's tolerance: ambiguous, not addable
    near = dict(JIRNY, id=3, name="Jirny garden", lat=50.1150)
    amb = oa.detect_location(_hdr(SITELAT=50.1, SITELONG=14.7), [JIRNY, near, DABLICE])
    assert amb.matched_id is None and amb.n_within == 2 and not amb.addable and "pick one" in amb.reasons[0]
    none = oa.detect_location(_hdr(INSTRUME="x"), [JIRNY])
    assert none.header_site is None and not none.addable and "no site in the header" in none.reasons[0]


def test_autodetect_from_source_reports_the_site(tmp_path: Path) -> None:
    from astropy.io import fits

    import optics_autodetect as oa

    src = tmp_path / "night"
    src.mkdir()
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.uint8),
                    header=_hdr(IMAGETYP="LIGHT", FOCALLEN=5480.0, APTDIA=800.0, SITELAT=50.112,
                                SITELONG=14.698)).writeto(src / "a.fits")
    rep = oa.autodetect_from_source(src, equipments=[], telescopes=[AZ800, ZEISS], locations=[JIRNY, DABLICE])
    assert rep.telescope.matched_id == 1 and rep.location.matched_id == 1
    assert not any(g["field"] == "Observer site" for g in rep.unresolved)


# ---------------------------------------------------------------------------- B. the run guard
def test_guard_blocks_a_selection_the_headers_contradict() -> None:
    import optics_autodetect as oa

    h = _c5a_az800_header(SITELAT=50.112, SITELONG=14.698)
    kw = dict(equipments=[C5A, QHY], telescopes=[AZ800, ZEISS])
    wrong = oa.rig_header_mismatches(h, equipment_id=2, telescope_id=2, header_path="/x/l1.fits", **kw)
    assert len(wrong) == 2
    assert "match camera 1: C5A-150M (c5a)" in wrong[0] and "2: QHY294MM (qhy)" in wrong[0] and "l1.fits" in wrong[0]
    assert "match telescope 1: AZ800 (az)" in wrong[1]
    assert oa.rig_header_mismatches(h, equipment_id=1, telescope_id=1, **kw) == []
    # the site: Dablice selected, the frames say Jirny (15 km) -> blocked; Jirny selected -> fine
    site = SimpleNamespace(name="Dablice", lat=DABLICE["lat"], lon=DABLICE["lon"])
    msgs = oa.rig_header_mismatches(h, equipment_id=1, telescope_id=1, site=site, **kw)
    assert len(msgs) == 1 and "km from the selected site Dablice" in msgs[0]
    ok_site = SimpleNamespace(name="Jirny", lat=JIRNY["lat"], lon=JIRNY["lon"])
    assert oa.rig_header_mismatches(h, equipment_id=1, telescope_id=1, site=ok_site, **kw) == []
    # missing header facts are no contradiction
    assert oa.rig_header_mismatches(_hdr(IMAGETYP="LIGHT"), equipment_id=2, telescope_id=2, site=site, **kw) == []
    # the selected row scoring high itself (a twin row scores higher) is no contradiction
    twin = dict(C5A, ID=3, ALIAS="twin", GAIN_ADU=0.8)
    assert oa.rig_header_mismatches(_c5a_az800_header(EGAIN=0.8), equipment_id=1, telescope_id=1,
                                    equipments=[C5A, twin], telescopes=[AZ800]) == []


@pytest.fixture
def rig_db(tmp_path_is_data_root: Path) -> dict:

    from database import VyvarDatabase

    root = tmp_path_is_data_root
    (root / "config.json").write_text(json.dumps({"gaia_db_path": ""}), encoding="ascii")
    db = VyvarDatabase(root / "vyvar.sqlite3")
    ids = {
        "c5a": db.insert_equipment("C5A-150M", "c5a", "CMOS", "14208*10656", 3.76),
        "qhy": db.insert_equipment("QHY294MM", "qhy", "CMOS", "4164*2796", 4.63),
        "az": db.insert_telescope("AZ800", "az", 800.0, 5480.0),
        "cz": db.insert_telescope("Carl-Zeiss 200mm", "cz", 200.0, 1000.0),
        "jirny": db.insert_location("Jirny", 50.112, 14.698, 275.0),
        "dablice": db.insert_location("Dablice", 50.131, 14.487, 280.0),
    }
    db.close()
    src = root / "night"
    (src / "lights").mkdir(parents=True)
    _write_c5a_light(src / "lights" / "l1.fits", SITELAT=50.112, SITELONG=14.698)
    return {"root": root, "src": src, **ids}


def _pre(rig_db: dict, *, eq: str, tel: str, loc: str, dry_run: bool = True):
    from config import AppConfig
    from database import VyvarDatabase
    from night_run import NightRunParams, check_run_preconditions

    db = VyvarDatabase(rig_db["root"] / "vyvar.sqlite3")
    try:
        p = NightRunParams(source_dir=rig_db["src"], equipment_id=rig_db[eq], telescope_id=rig_db[tel],
                           location_id=rig_db[loc], dry_run=dry_run)
        return check_run_preconditions(AppConfig(), p, db).problems
    finally:
        db.close()


def test_run_preconditions_refuse_a_rig_the_light_headers_contradict(rig_db: dict) -> None:
    """GUI (``_run_button_blockers``) and CLI (``run_night_pipeline``) both call check_run_preconditions."""
    probs = _pre(rig_db, eq="qhy", tel="cz", loc="dablice")
    assert sum("The light frames (l1.fits) match camera" in m for m in probs) == 1
    assert sum("match telescope" in m for m in probs) == 1
    assert sum("from the selected site Dablice" in m for m in probs) == 1
    assert _pre(rig_db, eq="c5a", tel="az", loc="jirny") == []


def test_incomplete_camera_stops_the_run_preconditions(rig_db: dict) -> None:
    """RIG-ADD-2026-11: a camera without gain / read noise (e.g. added from a header outside the catalogue)
    stops the run at its preconditions, not at MASTERSTAR."""
    probs = _pre(rig_db, eq="c5a", tel="az", loc="jirny", dry_run=False)
    assert any("Camera gain / read noise unknown" in m for m in probs), probs


def test_gui_run_button_is_blocked_by_the_guard(rig_db: dict, monkeypatch) -> None:
    from config import AppConfig
    from database import VyvarDatabase

    app = _src_app()
    monkeypatch.setattr(app, "_vyvar_job_running", lambda cfg: False)
    db = VyvarDatabase(rig_db["root"] / "vyvar.sqlite3")
    try:
        reasons = app._run_button_blockers(AppConfig(), db, source_root=str(rig_db["src"]),
                                           equipment_id=rig_db["qhy"], telescope_id=rig_db["az"],
                                           location_id=rig_db["jirny"])
    finally:
        db.close()
    assert any("match camera" in r and "QHY294MM" in r for r in reasons), reasons


def test_sample_header_is_remembered_until_the_file_changes(tmp_path: Path, monkeypatch) -> None:
    from astropy.io import fits

    import optics_autodetect as oa

    src = tmp_path / "s"
    src.mkdir()
    f = src / "a.fits"
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.uint8), header=_hdr(IMAGETYP="LIGHT", INSTRUME="A")).writeto(f)
    calls = []
    real = oa.find_sample_light_header
    monkeypatch.setattr(oa, "find_sample_light_header", lambda r, **k: calls.append(r) or real(r, **k))
    assert oa.sample_light_header(src)[0]["INSTRUME"] == "A"
    assert oa.sample_light_header(src)[0]["INSTRUME"] == "A" and len(calls) == 1
    fits.PrimaryHDU(np.zeros((3, 3), dtype=np.uint8), header=_hdr(IMAGETYP="LIGHT", INSTRUME="B")).writeto(
        f, overwrite=True)
    assert oa.sample_light_header(src)[0]["INSTRUME"] == "B" and len(calls) == 2


# ------------------------------------------------------------------------------ C. add a header rig to the DB
def test_prefill_takes_geometry_and_names_never_gain() -> None:
    import optics_autodetect as oa

    h = _c5a_az800_header(GAIN=100, EGAIN=0.8, RDNOISE=3.0, SATURATE=65535, XPIXSZ=15.04, SENSOR="CMOS",
                          SITELAT=50.112, SITELONG=14.698, SITEELEV=275.0, OBSERVAT="Jirny")
    pre = oa.header_rig_prefill(h)
    eq = pre["equipment"]
    assert eq["camera_name"] == "C5A-150M" and eq["sensor_size"] == "14208*10656" and eq["sensor_type"] == "CMOS"
    assert eq["pixel_um"] == pytest.approx(3.76)  # XPIXSZ is binned (4x)
    assert [m["key"] for m in eq["catalog_models"]] == ["moravian_c5a_150m"]
    flat = json.dumps(pre).lower()
    for banned in ("gain", "noise", "satur", "adc"):
        assert banned not in flat, banned
    assert pre["telescope"] == {"telescope_name": "AZ800", "focal_mm": 5480.0, "aperture_mm": 800.0}
    assert pre["location"] == {"name": "Jirny", "lat": 50.112, "lon": 14.698, "elev": 275.0}
    assert "location" not in oa.header_rig_prefill(_hdr(INSTRUME="x"))


def test_add_from_header_wrappers_fail_loud_and_insert(rig_db: dict) -> None:
    from database import VyvarDatabase

    db = VyvarDatabase(rig_db["root"] / "vyvar.sqlite3")
    try:
        with pytest.raises(ValueError, match="name is required"):
            db.add_equipment_from_header(camera_name=" ", alias="", sensor_type="", sensor_size="10*10",
                                         pixel_size=3.0)
        with pytest.raises(ValueError, match="Sensor size"):
            db.add_equipment_from_header(camera_name="X", alias="", sensor_type="", sensor_size="", pixel_size=3.0)
        with pytest.raises(ValueError, match="Pixel size"):
            db.add_equipment_from_header(camera_name="X", alias="", sensor_type="", sensor_size="10*10",
                                         pixel_size=0.0)
        # values the header does not state stay NULL (amended 2026-11-05: never invented, listed for the editor)
        assert db.location_altitude_m(db.add_location_from_header(place_name="N", latitude=49.0, longitude=16.0,
                                                                  altitude=None)) is None
        with pytest.raises(ValueError, match="Focal length"):
            db.add_telescope_from_header(telescope_name="T", alias="", diameter=200.0, focal=None)
        with pytest.raises(ValueError, match="elevation"):
            db.add_location_from_header(place_name="S", latitude=50.0, longitude=14.0, altitude="high")
        eid = db.add_equipment_from_header(camera_name="NewCam", alias="n", sensor_type="CMOS",
                                           sensor_size="1000*800", pixel_size=4.0)
        tid = db.add_telescope_from_header(telescope_name="NewTel", alias="t", diameter=150.0, focal=750.0)
        lid = db.add_location_from_header(place_name="NewSite", latitude=49.0, longitude=16.6, altitude="300")
        row = dict(db.conn.execute("SELECT * FROM EQUIPMENTS WHERE ID = ?;", (eid,)).fetchone())
        assert row["GAIN_ADU"] is None and row["READNOISE_E"] is None and row["SATURATE_ADU"] is None
        assert row["ACTIVE"] == "YES" and row["SENSORSIZE"] == "1000*800"
        assert db._fk_row_exists("TELESCOPE", tid) and db._fk_row_exists("LOCATION", lid)
    finally:
        db.close()


# --------------------------------------------------------------------------- GUI end to end (AppTest)
def _gui_env(tmp_path: Path, monkeypatch, *, default_site: str = "Dablice"):
    from vyvar_runtime import ensure_release_data_dir

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("VYVAR_DATA_DIR", str(root))
    ensure_release_data_dir(REPO)
    from config import AppConfig
    from database import VyvarDatabase

    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    db.insert_equipment("QHY294MM", "qhy", "CMOS", "4164*2796", 4.63)
    db.insert_telescope("Carl-Zeiss 200mm", "cz", 200.0, 1000.0)
    ids = {"Zdanice": db.insert_location("Zdanice", 49.07, 17.03, 300.0),
           "Dablice": db.insert_location("Dablice", 50.131, 14.487, 280.0)}
    db.set_reference_default("LOCATION", ids[default_site])
    db.conn.close()
    _src_app()
    from streamlit.testing.v1 import AppTest

    f = tmp_path / "app_main.py"
    f.write_text(APP_SCRIPT, encoding="utf-8")
    return cfg, ids, AppTest.from_file(str(f), default_timeout=120)


def _rows(cfg) -> dict:
    import sqlite3

    con = sqlite3.connect(cfg.database_path)
    try:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0] for t in ("EQUIPMENTS", "TELESCOPE", "LOCATION")}
    finally:
        con.close()


def test_gui_start_preselects_the_default_site_and_keeps_the_users_choice(tmp_path: Path, monkeypatch) -> None:
    """Milan 2026-11-05: on opening the GUI the DB default site is preselected (not the lowest ID); a later choice
    of the user survives a view switch."""
    _cfg, ids, at = _gui_env(tmp_path, monkeypatch)
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="vyvar_varstrem_location").value == f"{ids['Dablice']}: Dablice"
    at.selectbox(key="vyvar_varstrem_location").select(f"{ids['Zdanice']}: Zdanice").run()
    at.session_state["vyvar_main_view"] = "FITS QA"
    at.run()
    at.session_state["vyvar_main_view"] = "VAR-STREM"
    at.run()
    assert at.selectbox(key="vyvar_varstrem_location").value == f"{ids['Zdanice']}: Zdanice"


def test_gui_one_press_adds_missing_rig_and_site_and_selects_them(tmp_path: Path, monkeypatch) -> None:
    cfg, _ids, at = _gui_env(tmp_path, monkeypatch)
    src = tmp_path / "night"
    src.mkdir()
    _write_c5a_light(src / "l1.fits", SITELAT=50.112, SITELONG=14.698, SITEELEV=275.0, OBSERVAT="Jirny")
    at.run()
    at.text_input(key="vyvar_varstrem_source").input(str(src)).run()
    before = _rows(cfg)
    next(b for b in at.button if b.label == "Auto-detect optics from FITS").click().run()
    assert not at.exception, at.exception
    after = _rows(cfg)
    assert {t: after[t] - before[t] for t in after} == {"EQUIPMENTS": 1, "TELESCOPE": 1, "LOCATION": 1}
    assert ": C5A-150M (" in at.selectbox(key="vyvar_varstrem_equipment").value
    assert "AZ800" in at.selectbox(key="vyvar_varstrem_telescope").value
    assert at.selectbox(key="vyvar_varstrem_location").value.endswith(": Jirny")
    info = "\n".join(str(i.value) for i in at.info)
    assert "Added to DB" in info and "camera catalogue" in info and "focal 5480 mm, aperture 800 mm" in info
    assert "lat 50.112000, lon 14.698000, elevation 275 m" in info
    assert not [b for b in at.button if b.label.startswith("Add this")]  # one code path, no form step
    # pressed twice: matched by the same rules, nothing added again
    next(b for b in at.button if b.label == "Auto-detect optics from FITS").click().run()
    assert _rows(cfg) == after and not at.info


def test_header_without_names_or_elevation_gets_generated_names_and_blocks(rig_db: dict) -> None:
    """Generated names; the values the header does not state stay empty and RUN is blocked (camera without gain /
    read noise, site without elevation) - nothing invented."""
    import shutil

    from config import AppConfig
    from database import VyvarDatabase, get_observer_locations
    from night_run import NightRunParams, check_run_preconditions

    import optics_autodetect as oa

    shutil.rmtree(rig_db["src"] / "lights")
    (rig_db["src"] / "lights").mkdir()
    _write_c5a_light(rig_db["src"] / "lights" / "l1.fits", INSTRUME="", TELESCOP="", FOCALLEN=600.0, APTDIA="",
                     XPIXSZ=60.16, SITELAT=49.2, SITELONG=16.6)
    db = VyvarDatabase(rig_db["root"] / "vyvar.sqlite3")
    dbp = str(rig_db["root"] / "vyvar.sqlite3")
    try:
        db.conn.execute("DELETE FROM EQUIPMENTS WHERE ID = ?;", (rig_db["c5a"],))  # a camera the DB lacks
        db.conn.commit()
        eq, tel = db.autodetect_candidates()
        rep = oa.autodetect_from_source(rig_db["src"], equipments=eq, telescopes=tel,
                                        locations=get_observer_locations(dbp, active_only=True))
        notes = oa.add_header_rig_to_db(db, dbp, rep)
        assert rep.equipment.label.endswith(": Camera 14208x10656 3.76um ()")
        assert rep.telescope.label.endswith(": Telescope f=600 mm ()")
        assert rep.location.label.endswith(": SITE 49.2000 16.6000")
        text = "\n".join(notes)
        assert "INCOMPLETE" in text and "aperture EMPTY" in text and "elevation EMPTY" in text
        p = NightRunParams(source_dir=rig_db["src"], equipment_id=rep.equipment.matched_id,
                           telescope_id=rep.telescope.matched_id, location_id=rep.location.matched_id)
        probs = check_run_preconditions(AppConfig(), p, db).problems
        assert any("has no elevation" in m for m in probs), probs
        # second press: nothing added
        eq, tel = db.autodetect_candidates()
        rep2 = oa.autodetect_from_source(rig_db["src"], equipments=eq, telescopes=tel,
                                         locations=get_observer_locations(dbp, active_only=True))
        n_rows = [db.conn.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0] for t in ("EQUIPMENTS", "TELESCOPE",
                                                                                          "LOCATION")]
        notes2 = oa.add_header_rig_to_db(db, dbp, rep2)
        assert all("already in the DB" in n for n in notes2), notes2
        assert n_rows == [db.conn.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
                          for t in ("EQUIPMENTS", "TELESCOPE", "LOCATION")]
    finally:
        db.close()


# ------------------------------------------------------------------------------------------------ decisions
def test_decisions_record_the_change() -> None:
    text = (REPO / "docs" / "VYVAR_DECISIONS.md").read_text(encoding="utf-8")
    i = text.index("## S6-FIX-SITE-MANUAL-2026-10")
    assert "SUPERSEDED by RIG-AUTOSELECT-2026-11" in text[i:text.index("\n## ", i + 5)]
    for did in ("RIG-AUTOSELECT-2026-11", "SITE-MATCH-2026-11", "RIG-GUARD-2026-11", "RIG-ADD-2026-11"):
        assert f"## {did}" in text, did


# ------------------------------------------------- follow-up (2) sexagesimal header sites, LAT-OBS / LONG-OBS
@pytest.mark.parametrize("text,deg", [
    ("+50 05 30", 50.0 + 5 / 60 + 30 / 3600), ("50:05:30", 50.0 + 5 / 60 + 30 / 3600),
    ("-14 41 53.7", -(14 + 41 / 60 + 53.7 / 3600)), ("50d05m30s", 50.0 + 5 / 60 + 30 / 3600),
    ("14 41 53.7 W", -(14 + 41 / 60 + 53.7 / 3600)), ("50.112", 50.112), ("-0 30 00", -0.5),
])
def test_sexagesimal_site_texts_are_read(text: str, deg: float) -> None:
    from param_resolver import parse_angle_deg

    got = parse_angle_deg(text)
    assert got is not None and got[0] == pytest.approx(deg, abs=1e-12)


def test_header_site_reader_sexagesimal_keys_and_precision() -> None:
    from param_resolver import header_site_reading, resolve_site, site_match_tolerance_m

    h = _hdr(SITELAT="+50 06 43", SITELONG="14 41 53")
    hs = header_site_reading(h)
    assert hs.lat == pytest.approx(50.111944, abs=1e-6) and hs.lon == pytest.approx(14.698056, abs=1e-6)
    # rounding of whole arcseconds: 0.5" = 15.4 m (lat), 9.9 m (lon)
    assert site_match_tolerance_m(hs) == pytest.approx(1000.0 + np.hypot(15.44, 9.92), abs=0.2)
    # whole arcminutes: 0.5' = 927 m in latitude
    assert site_match_tolerance_m(header_site_reading(_hdr(SITELAT="50 07", SITELONG="14 42"))) > 1900.0
    lo = header_site_reading(_hdr(**{"LAT-OBS": 50.112, "LONG-OBS": 14.698, "ALT-OBS": 275.0}))
    assert (lo.lat_key, lo.lon_key, lo.elev) == ("LAT-OBS", "LONG-OBS", 275.0)
    # resolve_site reads the same: sexagesimal header site without a draft
    r = resolve_site(_hdr(SITELAT="50:06:43", SITELONG="14:41:53", SITEELEV=275.0), allow_config=False)
    assert r.ok and r.source == "header" and r.lat == pytest.approx(50.111944, abs=1e-6)


def test_resolve_site_cross_check_uses_the_same_site_rule(monkeypatch) -> None:
    """The old 0.01 deg box warned for a site 0.012 deg (1.3 km) away in latitude only above 0.01 deg and
    missed a 1.3 km offset split over both axes; now one rule: distance > SITE-MATCH tolerance."""
    import param_resolver as pr

    monkeypatch.setattr(pr, "_draft_location", lambda db, did: (50.112, 14.698, 275.0))
    near = pr.resolve_site(_hdr(SITELAT=50.1125, SITELONG=14.6985), db=object(), draft_id=1, allow_config=False)
    assert not any("disagrees" in w for w in near.warnings)  # 66 m: the same site
    # 0.009 deg in both axes = 1.17 km: beyond 1000 m + rounding, inside the old 0.01 deg box
    far = pr.resolve_site(_hdr(SITELAT=50.121, SITELONG=14.707), db=object(), draft_id=1, allow_config=False)
    assert any("disagrees" in w and "same-site tolerance" in w for w in far.warnings), far.warnings



# ------------------------------------------------------------- follow-up (4) bounded sample-header search
def _fits(path: Path, imagetyp: str) -> None:
    from astropy.io import fits

    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.uint8), header=_hdr(IMAGETYP=imagetyp)).writeto(path)


def _count_scandir(monkeypatch) -> list:
    import os

    seen: list = []
    real = os.scandir
    monkeypatch.setattr(os, "scandir", lambda p=".": seen.append(str(p)) or real(p))
    return seen


def test_sample_search_stops_at_the_first_light_breadth_first(tmp_path: Path, monkeypatch) -> None:
    """A deep calibration branch (60 levels, a dark at each) sorts before the light folder; the old search
    listed the whole tree and read every dark header first. Now: light folders first, stop at the first light."""
    import optics_autodetect as oa

    src = tmp_path / "night"
    deep = src / "Calib"
    for i in range(60):
        deep = deep / f"d{i:02d}"
        _fits(deep / "dark.fits", "DARK")
    _fits(src / "Lights" / "V" / "l1.fits", "LIGHT")
    seen = _count_scandir(monkeypatch)
    stats: dict = {}
    hdr, path = oa.find_sample_light_header(src, stats=stats)
    assert path is not None and path.endswith("l1.fits") and hdr["IMAGETYP"] == "LIGHT"
    assert stats["headers"] == 1 and stats["dirs"] <= 4 and len(seen) == stats["dirs"], (stats, seen)


def test_sample_search_is_bounded_and_says_so(tmp_path: Path, monkeypatch) -> None:
    import optics_autodetect as oa

    src = tmp_path / "archive"
    for i in range(30):
        for j in range(30):
            (src / f"n{i:02d}" / f"s{j:02d}").mkdir(parents=True)
    _fits(src / "n00" / "dark.fits", "DARK")
    seen = _count_scandir(monkeypatch)
    stats: dict = {}
    assert oa.find_sample_light_header(src, stats=stats) == (None, None)
    assert stats["exhausted"] and stats["dirs"] == oa.SAMPLE_SCAN_MAX_DIRS == len(seen)  # not all 931 folders
    rep = oa.autodetect_from_source(src, equipments=[], telescopes=[])
    assert rep.header_path is None and "No light frame" in rep.no_light and "search bound reached" in rep.no_light


def test_run_preconditions_say_when_no_light_is_found(rig_db: dict) -> None:
    import shutil

    shutil.rmtree(rig_db["src"] / "lights")
    (rig_db["src"] / "lights").mkdir()
    probs = _pre(rig_db, eq="c5a", tel="az", loc="jirny")
    assert any(m.startswith("No light frame") and "cannot be checked against the frames" in m for m in probs), probs


# ------------------------------------------------- follow-up (1) the blocker says exactly how to resolve it
def test_blocker_messages_say_how_to_resolve() -> None:
    import optics_autodetect as oa

    h = _c5a_az800_header(SITELAT=50.112, SITELONG=14.698)
    site = SimpleNamespace(name="Dablice", lat=DABLICE["lat"], lon=DABLICE["lon"])
    cam, tel, loc = oa.rig_header_mismatches(h, equipment_id=2, telescope_id=2, site=site,
                                             equipments=[C5A, QHY], telescopes=[AZ800, ZEISS])
    assert "To resolve: select 1: C5A-150M (c5a)" in cam and "--camera 1" in cam and "correct its DB row" in cam
    assert "--telescope 1" in tel and "FOCALLEN" in tel
    for part in ("To resolve: select the header site", "--site", "Add this site from the header to the DB",
                 "correct SITELAT/SITELONG in the FITS headers"):
        assert part in loc, part
