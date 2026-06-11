# VYVAR -- Gaia DR3 integration audit

**Started:** 2026-06-10 -- **Auditor:** Claude (read-only) -- **Status:** GAIA-1/GAIA-2 deferred
to DR4 (2026-06-10, Milan); GAIA-3 closed; no code changed.

**Scope:** Gaia DR3 ingest path only -- catalog build -> on-disk schema -> detection match
(Gaia <-> DAO) -> comparison reference magnitude. NOT the blind plate-solver index
(that reads only ra/dec/g_mag and is unaffected by every finding below).

**Method:** code read of the cloned repo (`origin/main` @ `39690b7`) cross-checked against the
uploaded `build_gaia_catalog.py` / `build_blind_index.py` (verified byte-identical to repo via
`diff`), plus Gaia DR3 Documentation release 1.3
(`https://gea.esac.esa.int/archive/documentation/GDR3/`). DR3 reference epoch = J2016.0
(astrometry carried over from EDR3); observations are 2026 (~10 yr baseline).

**Severity legend:** HIGH / MEDIUM / LOW / INFO. "Build-decision" = must be settled before the
in-progress full-sky rebuild finishes, because the cheap fix is a column in the first TAP pass.

---

## Verdict summary

| ID | Finding | Severity | Build-decision | Verdict |
|----|---------|----------|----------------|---------|
| GAIA-1 | `pmra`/`pmdec` absent from catalog; platesolver PM propagation silently no-ops | MEDIUM (HIGH on fine+dense) | YES | Add columns -- recommended |
| GAIA-2 | No astrometric-quality filter (`ruwe`, `duplicated_source`) ingested | MEDIUM | YES | Add `ruwe` (+ optional `duplicated_source`) -- recommended |
| GAIA-3 | Riello G-band magnitude correction not applied | -- | NO | RESOLVED by DR3 docs: already baked into DR3 values; must NOT re-apply. Close. |
| GAIA-4 | Comp reference magnitude = Gaia G, no transform to observed filter | LOW (for build) | NO | KNOWN / already tracked (APCORR-COLOR parked, AAVSO-standard open). Downstream, not a column. |
| GAIA-5 | `phot_bp_rp_excess_factor` (C*) not ingested | LOW | OPTIONAL | Only matters if BP/RP flux is used for science; VYVAR uses `bp_rp` for colour tier only. Optional add. |
| GAIA-6 | Coordinate frame + match-tolerance vs epoch offset | INFO | NO | Confirmed consistent (ICRS @ 2016.0); tolerance absorbs PM on wide; fine+dense is the GAIA-1 risk. |

**Net build action:** the only catalog-schema changes worth a rebuild restart are GAIA-1
(`pmra`, `pmdec`) and GAIA-2 (`ruwe`, optionally `duplicated_source`). Everything else is either
resolved, downstream, or optional and non-blocking.

---

## Findings (detail)

### GAIA-1 -- Proper motion / epoch propagation (MEDIUM; HIGH on fine + dense fields)

**Observation.** The catalog build SELECT does not request `pmra`/`pmdec` in either branch:
full `gaiadr3.gaia_source` (`build_gaia_catalog.py` ~328-345, table at :346) and lite
`gaiadr3.gaia_source_lite` (~353-365, table at :366). `_ROW_COLUMNS` (:72-89) and the
`init_db` schema (:374-393) likewise omit them. So the on-disk `gaia_dr3` table has no PM.

The downstream propagation logic, however, *exists and is correct*:
`vyvar_platesolver.py` defines `GAIA_EPOCH = 2016.0` (:63), `PM_CORRECTION_MIN_MASYR = 10.0`
(:64), and `_apply_proper_motion(...)` (:67-96) which advances ra/dec from 2016.0 to the
observation year. It is invoked per source (:124-136) and logs the corrected count (:2858).
The catch: it reads `rr.get("pmra")` / `rr.get("pmdec")` and falls back to `0.0` when absent
(:124-130). Against the main catalog those keys are always missing -> the correction silently
becomes a no-op. (`database.py` :210-212 already treats `pmra`/`pmdec` as *optional* columns,
and `scripts/chiandh_build_field_db.py` :41 *does* pull them for per-field DBs -- so the wiring
expects them; only the main full-sky build omits them.)

**DR3 reference.** Reference epoch J2016.0. PM is available for 5-parameter (585.4M) and
6-parameter (882.3M) solutions; 2-parameter solutions (344.0M, ~19% of 1.81B) carry position
only -> `pmra`/`pmdec` NULL. Bright comparison stars (G <= 16.5) are overwhelmingly 5/6-param,
so PM is available where it matters; NULL rows simply skip propagation (correct fallback).

**Impact.** Wide rig (~9.77 arcsec/px): negligible -- see GAIA-6, tolerance dwarfs any PM shift.
Fine rig (Newton ~0.65 arcsec/px): a star at e.g. 200 mas/yr moves ~2 arcsec over 10 yr. That
will not push it outside the match radius by itself, but in a dense field (h & chi Per,
globulars) where a *wrong* Gaia neighbour sits within the tolerance, the shift can flip the
nearest-match association -> wrong `source_id` and wrong reference `g_mag` assigned to the
detection. This is the classic "silent wrong value" class, not a crash.

**Recommendation.** Add `pmra`, `pmdec` to both SELECT branches, to `_ROW_COLUMNS`, and to the
`init_db` schema (REAL, nullable). No downstream change needed -- the platesolver immediately
starts propagating once the keys are present. Cheapest in the first TAP pass (a few bytes/row).

**Verdict:** add columns. Build-decision = YES.

---

### GAIA-2 -- Astrometric-quality filter not ingested (MEDIUM)

**Observation.** `ruwe` and `duplicated_source` are pulled by neither SELECT branch and are read
nowhere in production (repo-wide grep: 0 hits outside `scripts/archive`). The build does ingest
two partial-quality fields -- `phot_variable_flag AS var_flag` and `non_single_star`
(`build_gaia_catalog.py` :344-345, :364-365) -- but not the primary astrometric-quality
indicator.

**DR3 reference.** `ruwe` (renormalised unit weight error) is the standard single-star
astrometric-quality gate; values near 1.0 indicate a well-behaved single-star solution, with the
commonly used cut at roughly <= 1.4. `duplicated_source` flags observational/processing
ambiguity (e.g. a close pair partially resolved across transits).

**Impact.** A comparison star with poor `ruwe` has an unreliable catalog position -> elevated
mis-match risk (compounds GAIA-1 on fine scale) and a potentially blended/duplicated source used
as a photometric reference. For journal-grade differential photometry the comp ensemble should
be filterable on `ruwe`.

**Recommendation.** Add `ruwe` to both SELECT branches + `_ROW_COLUMNS` + schema (REAL). Add
`duplicated_source` (INTEGER) if cheap. Filtering/threshold wiring is a *separate, later* code
task (no urgency); only the *ingest* is build-coupled. Suggest ingest now, gate later.

**Verdict:** add `ruwe` (+ optional `duplicated_source`). Build-decision = YES.

---

### GAIA-3 -- Riello G-band magnitude correction (RESOLVED -- close)

**Prior concern (2026-06-09, from memory):** that the EDR3/Riello et al. 2021 G-band correction
for 6-parameter and 2-parameter solutions was not applied.

**DR3 reference (decisive).** The DR3 documentation states the milli-magnitude G-band correction
for 6-param and 2-param solutions is already included in the DR3 archive values and should NOT be
applied again to broad-band photometry taken from the DR3 tables. (DR3 doc landing page,
photometry note; ref. Riello et al. 2021.)

**Verdict.** Not a gap. VYVAR is correct to leave `phot_g_mean_mag` untouched. Applying the
correction would double-count it. CLOSE. (This supersedes the earlier "candidate" item.)

---

### GAIA-4 -- Colour / filter system of the reference magnitude (KNOWN -- not a column)

**Observation.** The comparison reference magnitude is Gaia G (`comp_selection_per_target.py`
:142, :1223 -- `g_mag` / `phot_g_mean_mag` / `catalog_mag`); BP-RP drives the colour tier only.
A Riello-2021 BP-RP -> Johnson B-V transform exists but is informational / report-side only
(`photometry_report.py` :384; `citations.py` :182 "Johnson B-V retired", :250 emits
`riello2021`). There is no transform of the reference magnitude into the observed filter (B/V/Rc).

**Impact.** For the *differential* path most of this cancels in the local-comp ensemble. It
matters for absolute placement on a standard system -- which is the paper's calibration concern,
already tracked: APCORR-COLOR is PARKED and the "AAVSO-standard output (G -> B/V/Rc or APASS)"
item is OPEN in the ROADMAP. The DR3 route for absolute work is synthetic photometry
(`gaiadr3.synthetic_photometry_gspc`, Gaia Collaboration 2022h) -- a downstream option, not a
catalog column.

**Verdict.** Known and tracked; no new action; not coupled to the build. LOW.

---

### GAIA-5 -- BP/RP excess factor C* not ingested (LOW -- optional)

**Observation.** `phot_bp_rp_excess_factor` (and its colour-corrected form C*) is not pulled.
**Impact.** It is a contamination/reliability indicator for BP/RP *fluxes* (crowding, binarity).
VYVAR consumes `bp_rp` only as a colour for tiering, not as a science flux, so the practical
impact is small. **Recommendation.** Optional ingest if you ever want to QC the colour used for
tiering in crowded fields; otherwise skip. Not build-blocking. LOW.

---

### GAIA-6 -- Coordinate frame + match tolerance vs epoch offset (INFO -- confirmed)

**Observation.** The catalog stores `ra`/`dec` straight from `gaiadr3.gaia_source[_lite]` =
ICRS at reference epoch J2016.0, consistent with the platesolver's `GAIA_EPOCH = 2016.0` (:63).
Match tolerance is scale-aware: `per_frame_catalog_match_sep_arcsec_for_scale` (`utils.py`
:381-390) = ~2.5 px in arcsec, floor 3.0 arcsec, fallback 20.0 arcsec; the pixel radius
`_catalog_match_radius_px` (`pipeline.py` :6177-6193) has a 10 px floor; default
`match_sep_arcsec = 8.0` (:6385, :6606). Match provenance is recorded
(`gaia_match_arcsec`/`_quality`/`_source`, :5675-5677; `gaia_matches_within_10arcsec`, :6006).

**Assessment.** Wide rig: ~2.5 px ~= 24 arcsec tolerance -> any plausible 10-yr PM shift is
negligible; epoch offset is safely absorbed. Fine rig: effective radius ~3-6.5 arcsec; the
*magnitude* tolerance is fine, but in dense fields the unpropagated PM (GAIA-1) is the real
mis-association risk. No frame inconsistency found.

**Verdict.** Consistent; folds into GAIA-1 for the fine+dense case. INFO.

---

## Decision required while the rebuild runs

The full-sky rebuild was at strip 133/713 (~19%), ETA ~50 h, ~28.8M projected inserts.

- Adding GAIA-1 + GAIA-2 columns to the **first** TAP pass is nearly free (marginal bytes/row).
- Deferring means either a second ~50 h pass or an `ALTER TABLE ADD COLUMN` + per-`source_id`
  UPDATE backfill -- and the backfill re-pays the same per-strip TAP latency, so it is not cheap.
- At 19% done, restarting now sacrifices ~10 h of work vs. risking a ~50 h redo later.

**Recommendation:** add `pmra`, `pmdec`, `ruwe` (and optionally `duplicated_source`) to the
SELECT + `_ROW_COLUMNS` + `init_db` schema and restart now.

**Caveat (verify first):** the build defaults to `gaiadr3.gaia_source_lite`
(`build_gaia_catalog.py` :228, :366). Confirm the lite table exposes `pmra`/`pmdec`/`ruwe`
(`SELECT TOP 1 pmra, pmdec, ruwe FROM gaiadr3.gaia_source_lite`). If any is missing there,
switch to `--full-source` (`gaiadr3.gaia_source`) -- slower and more prone to TAP 500s, but it
has the full column set.

## Implementation note (for Cursor -- not done here)

Changes are confined to `GAIA_DR3/build_gaia_catalog.py`: extend both ADQL SELECT branches,
`_ROW_COLUMNS` (:72-89), the `init_db` CREATE TABLE (:374-393), and the INSERT column/placeholder
strings (:534-535) consistently. Downstream readers already tolerate the new columns
(`database.py` :210-212 optional-column handling). No change needed to `build_blind_index.py`
(reads ra/dec/g_mag only). Keep file content ASCII-only per project discipline.

## Disposition

- GAIA-1, GAIA-2: **deferred to Gaia DR4 build** (~Dec 2026) as of 2026-06-10. DR3 rebuild
  completes on existing schema. See `VYVAR_DECISIONS.md` (DR4 migration hooks).
- GAIA-3: closed (resolved by DR3 docs).
- GAIA-4: no action (tracked elsewhere; APCORR-COLOR / AAVSO-standard).
- GAIA-5: optional, non-blocking.
- GAIA-6: informational, confirmed.

*Supersedes the "await build-decision" disposition on GAIA-1/GAIA-2 recorded at audit filing.*
