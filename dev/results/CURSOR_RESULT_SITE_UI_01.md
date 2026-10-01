CURSOR RESULT - 2026-10-01 SITE-UI-01

What I did
Fixed the defect where UI showed one observing site (e.g. Jirny) while RUN
used stale config (e.g. Zdanice) and logged source=ui_selection falsely.
UI RUN now passes the selectbox location_id; preselect prefers config then
IS_DEFAULT; persist to config.json happens on RUN only; RUN blocked without
a site; caption shows the site that will be used. Resolver precedence is
explicit > manifest > config with truthful source=. Manifest rig.location_id
updates when the explicit site differs.

## Defect cite (5697ba7)
- app.py:224-237 used cfg.observer_location_id with location_source_hint=
  "ui_selection" (not the selectbox).
- app.py:1630-1645 preselected IS_DEFAULT over config.
- app.py:1712-1730 CONFIG-WRITE-GUARD persisted only on selectbox *change*,
  so display could diverge from config silently.

## Fix summary
- src_py/observer_location.py: location_preselect_id; manifest tier;
  source includes "manifest".
- src_py/app.py: selectbox id -> NightRunParams.location_id; caption;
  RUN gate; persist on RUN.
- src_py/night_run.py: resolve_night_run_cli_ids returns loc_source;
  NightRunParams.manifest_location_id; site logged at import start.
- src_py/draft_provenance.py: update rig.location_id + log
  "[SITE] manifest location changed old -> new".

## Camera / telescope audit
PRESENT correct (no defect class). Live selectbox -> NightRunParams.
See session_20261001_site_ui_01/optics_audit.txt.
  Camera: app.py selectbox -> parse_ui_optics_from_labels -> equipment_id.
  Telescope: same for telescope_id.

## Tests
T1-T4 in dev/tests/test_site_ui_01.py: PASS (5 tests including preselect).

## Verification (real data)
1. Inventory: session_20261001_site_ui_01/draft_site_inventory.txt
   draft_000521: manifest location_id=5 Zdanice; [SITE] lines claim
   source=ui_selection (pre-fix evidence of the bug).
2. UI vs night_run parity for site=Jirny (id=2):
   ui_cli_resolve_parity.txt - both resolve to Jirny with identical
   lat/lon; NightRunParams.location_id=2 for both paths. Full dual night
   re-run of 521 (UI Streamlit + night_run.py on fresh copies) not
   executed in this session (multi-hour); architecture guarantees
   identical photometry when both pass the same location_id into
   run_night_pipeline. Milan may overnight-confirm hashes.
3. Airmass (analytic recompute from field center + LC JD):
   airmass_site_delta.txt
   V1023 Her / V1022 Her / V1148 Her: median airmass delta
   (Jirny - Zdanice) about -0.001 to -0.006. Stored LC airmass matches
   the Zdanice night. Detrended LC RMS after a full Jirny re-run is
   not remeasured here (needs the overnight re-run).

## Gates
- --fast --clean OVERALL PASS (1656 pytest).
- --full-epsf OVERALL PASS (~18910 s): aperture 87197716 / dd92e99d;
  epsf01 c94cf4fe; G3 dem 15.372/5.360 n_full=134. Aperture/epsf anchors
  unchanged by SITE-UI-01.

## Commits
- fix+tests: b6ff5e5
- result: 6a0f3ed

## STOP
Pushed alpha-fixes-01 only. main untouched.
